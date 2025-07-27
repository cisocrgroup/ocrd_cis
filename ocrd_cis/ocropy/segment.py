from __future__ import absolute_import

from typing import Optional
from logging import Logger
import itertools

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage import draw
from skimage.morphology import convex_hull_image
import cv2
from shapely.geometry import Polygon, LineString
from shapely.prepared import prep
from shapely.ops import unary_union, nearest_points
from shapely.validation import explain_validity
from shapely import set_precision

from ocrd_utils import (
    coordinates_of_segment,
    coordinates_for_segment,
    points_from_polygon,
    polygon_from_points,
)
from ocrd_models.ocrd_page import (
    CoordsType,
    TextLineType,
    TextRegionType,
    SeparatorRegionType,
    PageType,
    AlternativeImageType,
    OcrdPage
)
from ocrd_models.ocrd_page_generateds import (
    BaselineType,
    TableRegionType,
    ImageRegionType,
    RegionRefType,
    RegionRefIndexedType,
    OrderedGroupType,
    OrderedGroupIndexedType,
    UnorderedGroupType,
    UnorderedGroupIndexedType,
    ReadingOrderType
)
from ocrd import Processor
from ocrd.processor import OcrdPageResult, OcrdPageResultImage

from .ocrolib import midrange
from .ocrolib import morph
from .common import (
    pil2array,
    array2pil,
    check_page, check_region,
    determine_zoom,
    hmerge_line_seeds,
    compute_segmentation,
    lines2regions
)


def masks2polygons(logger: Logger, bg_labels, baselines, fg_bin, name, min_area=None, simplify=None, open_holes=False,
                   reorder=True):
    """Convert label masks into polygon coordinates.

    Given a Numpy array of background labels ``bg_labels``,
    (optionally) a Numpy array of a scalar field ``baselines``,
    and a Numpy array of the foreground ``fg_bin``,
    iterate through all labels (except zero and those labels
    which do not correspond to any foreground at all) to find
    their outer contours and inner baselines. 
    Each contour part which is not too small and gives a
    (simplified) polygon of at least 4 points becomes a polygon.
    (Thus, labels can be split into multiple polygons.)

    Return a tuple:
    - these polygons as a list of label, polygon, baseline tuples, and
    - a Numpy array of new background labels for that list.
    """
    # find sharp baseline
    if baselines is not None:
        def getx(xy):
            return xy[0]

        baselines = [LineString(sorted([p[::-1] for p in line], key=getx)).simplify(5)
                     for line in baselines
                     if len(line) >= 2]
    results = list()
    result_labels = np.zeros_like(bg_labels, dtype=bg_labels.dtype)
    for label in np.unique(bg_labels):
        if not label:
            # ignore if background
            continue
        bg_mask = np.array(bg_labels == label, bool)
        if not np.count_nonzero(bg_mask * fg_bin):
            # ignore if missing foreground
            logger.debug(f'Skipping label {label} in {name} due to empty fg')
            continue
        # simplify to convex hull
        if simplify is not None:
            hull = convex_hull_image(bg_mask.astype(np.uint8)).astype(bool)
            conflicts = np.setdiff1d(hull * simplify, bg_mask * simplify)
            if conflicts.any():
                logger.debug(
                    f'Cannot simplify {label}: convex hull would create additional intersections {str(conflicts)}')
            else:
                bg_mask = hull
        if open_holes:
            # def plot_poly(contour, color):
            #     import matplotlib.pyplot as plt
            #     from matplotlib.patches import Polygon as PolygonPatch
            #     plt.figure()
            #     plt.imshow(fg_bin)
            #     plt.gca().scatter(*zip(*contour[:,0]))
            #     plt.gca().add_patch(PolygonPatch(contour[:,0], alpha=0.5, color=color, closed=False))
            #     plt.show()
            # find outer contour (parts) plus direct holes (if any)
            contours = []
            cont, hier = cv2.findContours(bg_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            idx = 0
            while idx >= 0:
                contour = cont[idx]
                if len(contour) < 3:
                    idx = hier[0, idx, 0]
                    continue
                #plot_poly(contour, 'red')
                idx_hole = hier[0, idx, 2]
                while idx_hole >= 0:
                    hole = cont[idx_hole]
                    if len(hole) < 3:
                        idx_hole = hier[0, idx_hole, 0]
                        continue
                    logger.debug(
                        f"Label {label} contour {idx} [{len(contour)} pts] has hole {idx_hole} [{len(hole)} pts]")
                    #plot_poly(hole, 'blue')
                    # cut child from outside...
                    # first get nearest point on child
                    hole_idx = np.argmin([cv2.pointPolygonTest(contour, tuple(pt[0].tolist()), True)
                                          for pt in hole])
                    # now get nearest point on parent
                    # (we cannot use PolygonTest directly, because we must also interpolate
                    #  to prevent crossing edges; at least each 10px)
                    contour = np.append(contour, contour[0:1], axis=0)
                    contour2 = np.diff(contour, axis=0)
                    contourtics = np.maximum(1, np.linalg.norm(contour2, axis=2).astype(int)[:,0] // 10)
                    interpol = []
                    for i, ntics in enumerate(contourtics):
                        interpol.extend(np.array(
                            contour[i:i + 1] +
                            contour2[i:i + 1] *
                            np.linspace(0, 1, ntics)[:, np.newaxis, np.newaxis], int))
                    interpol.append(contour[-1])
                    interpol = np.array(interpol)
                    contourtics = np.insert(np.cumsum(contourtics), 0, 0)
                    assert np.all(contour == interpol[contourtics])
                    interpol_idx = np.linalg.norm(interpol - hole[hole_idx], axis=2).argmin()
                    contour_idx = np.searchsorted(contourtics, interpol_idx)
                    if interpol_idx in contourtics:
                        contour_idx2 = contour_idx + 1
                    else:
                        contour_idx2 = contour_idx
                    if contour_idx2 >= len(contour):
                        contour_idx2 = 0
                    cispoint1 = cispoint2 = interpol[interpol_idx:interpol_idx + 1]
                    if interpol_idx == 0:
                        diff1 = (interpol[-1:] - cispoint1) // 5
                    else:
                        diff1 = (interpol[interpol_idx - 1: interpol_idx] - cispoint1) // 5
                    if interpol_idx + 1 >= len(interpol):
                        diff2 = (interpol[0:1] - cispoint2) // 5
                    else:
                        diff2 = (interpol[interpol_idx + 1: interpol_idx + 2] - cispoint2) // 5
                    cispoint1 = cispoint1 + diff1
                    cispoint2 = cispoint2 + diff2
                    logger.debug(f"Stitching at interpolation pos {interpol_idx} hole pos {hole_idx}")
                    # now stitch together outer (up to cision), inner (re-arranged around cision), outer (rest)
                    # (this works, because inner contours have inverse direction)
                    contour = np.concatenate(
                        [contour[:contour_idx], cispoint1,
                         hole[hole_idx:], hole[:hole_idx],
                         cispoint2, contour[contour_idx:]])
                    #plot_poly(contour, 'green')
                    idx_hole = hier[0, idx_hole, 0]
                #plot_poly(contour, 'red')
                logger.debug(f"Adding label {label} contour {idx} [{len(contour)} pts]")
                contours.append(contour)
                idx = hier[0, idx, 0]
        else:
            # find outer contour (parts):
            contours, _ = cv2.findContours(bg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # determine areas of parts:
        areas = [cv2.contourArea(contour) for contour in contours]
        total_area = sum(areas)
        if not total_area:
            # ignore if too small
            continue
        # redraw label array
        contour_labels = np.zeros_like(bg_mask, np.uint8)
        for i, contour in enumerate(contours):
            cv2.drawContours(contour_labels, contours, i, i+1, cv2.FILLED)
        if reorder:
            # sort contours in reading order
            order = np.argsort(morph.reading_order(contour_labels)[1:])
        else:
            order = range(len(contours))
        # convert to polygons
        for i in order:
            contour = contours[i]
            area = areas[i]
            if min_area and area < min_area and area / total_area < 0.1:
                logger.warning(f'Label {label} contour {i} is too small ({area}/{total_area}) in {name}')
                continue
            # simplify shape:
            # can produce invalid (self-intersecting) polygons:
            #polygon = cv2.approxPolyDP(contour, 2, False)[:, 0, ::] # already ordered x,y
            polygon = contour[:, 0, ::]  # already ordered x,y
            # simplify and validate:
            polygon = Polygon(polygon)
            if not polygon.is_valid:
                #logger.debug(polygon.wkt)
                logger.debug(explain_validity(polygon))
            polygon = make_valid(polygon)
            if not polygon.is_valid:
                #LOG.debug(polygon.wkt)
                logger.warning(explain_validity(polygon))
            poly = polygon.exterior.coords[:-1]  # keep open
            if len(poly) < 4:
                logger.warning(f'Label {label} contour {i} for {name} has less than 4 points')
                continue
            # get baseline segments intersecting with this line mask
            # and concatenate them from left to right
            if baselines is not None:
                base = join_baselines(
                    logger,
                    [baseline.intersection(polygon) for baseline in baselines if baseline.intersects(polygon)], name)
                if base is not None:
                    base = base.coords
            else:
                base = None
            results.append((label, poly, base))
            result_labels[contour_labels == i + 1] = len(results)
    return results, result_labels


class OcropySegment(Processor):
    @property
    def executable(self):
        return 'ocrd-cis-ocropy-segment'

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Segment pages into regions+lines, tables into cells+lines, or regions into lines.

        Open and deserialise PAGE input file and its respective images,
        then iterate over the element hierarchy down to the requested level.

        \b
        Depending on ``level-of-operation``, consider existing segments:
        - If ``overwrite_separators=True`` on ``page`` level, then
          delete any SeparatorRegions.
        - If ``overwrite_regions=True`` on ``page`` level, then
          delete any top-level TextRegions (along with ReadingOrder).
        - If ``overwrite_regions=True`` on ``table`` level, then
          delete any TextRegions in TableRegions (along with their OrderGroup).
        - If ``overwrite_lines=True`` on ``region`` level, then
          delete any TextLines in TextRegions.
        - If ``overwrite_order=True`` on ``page`` or ``table`` level, then
          delete the reading order OrderedGroup entry corresponding
          to the (page/table) segment.

        Next, get each element image according to the layout annotation (from
        the alternative image of the page/region, or by cropping via coordinates
        into the higher-level image) in binarized form, and represent it as an array
        with non-text regions and (remaining) text neighbours suppressed.

        \b
        Then compute a text line segmentation for that array (as a label mask).
        When ``level-of-operation`` is ``page`` or ``table``, this also entails
        detecting
        - up to ``maximages`` large foreground images,
        - up to ``maxseps`` foreground line separators and
        - up to ``maxcolseps`` background column separators
        before text line segmentation itself, as well as aggregating text lines
        to text regions afterwards.

        Text regions are detected via a hybrid variant recursive X-Y cut algorithm
        (RXYC): RXYC partitions the binarized image in top-down manner by detecting
        horizontal or vertical gaps. This implementation uses the bottom-up text line
        segmentation to guide the search, and also uses both pre-existing and newly
        detected separators to alternatively partition the respective boxes into
        non-rectangular parts.

        During line segmentation, suppress the foreground of all previously annotated
        regions (of any kind) and lines, except if just removed due to ``overwrite``.
        During region aggregation however, combine the existing separators with the
        new-found separators to guide the column search.

        All detected segments (both text line and text region) are sorted according
        to their reading order (assuming a top-to-bottom, left-to-right ordering).
        When ``level-of-operation`` is ``page``, prefer vertical (column-first)
        succession of regions. When it is ``table``, prefer horizontal (row-first)
        succession of cells.

        \b
        Then for each resulting segment label, convert its background mask into
        polygon outlines by finding the outer contours consistent with the element's
        polygon outline. Annotate the result by adding it as a new TextLine/TextRegion:
        - If ``level-of-operation`` is ``region``, then append the new lines to the
          parent region.
        - If it is ``table``, then append the new lines to their respective regions,
          and append the new regions to the parent table.
          (Also, create an OrderedGroup for it as the parent's RegionRef.)
        - If it is ``page``, then append the new lines to their respective regions,
          and append the new regions to the page.
          (Also, create an OrderedGroup for it in the ReadingOrder.)

        Produce a new output file by serialising the resulting hierarchy.
        """
        # FIXME: allow passing a-priori info on reading order / textline order
        # (and then pass on as ``bt`` and ``rl``; however, there may be a mixture
        #  of different scripts; also, vertical writing needs internal rotation
        #  because our line segmentation only works for horizontal writing)
        overwrite_lines = self.parameter['overwrite_lines']
        overwrite_regions = self.parameter['overwrite_regions']
        overwrite_separators = self.parameter['overwrite_separators']
        overwrite_order = self.parameter['overwrite_order']
        oplevel = self.parameter['level-of-operation']

        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()

        # TODO: also allow grayscale_normalized (try/except?)
        page_image, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id, feature_selector='binarized')
        zoom = determine_zoom(self.logger, page_id, self.parameter['dpi'], page_image_info)

        # aggregate existing regions so their foreground can be ignored
        ignore = (page.get_ImageRegion() +
                  page.get_LineDrawingRegion() +
                  page.get_GraphicRegion() +
                  page.get_ChartRegion() +
                  page.get_MapRegion() +
                  page.get_MathsRegion() +
                  page.get_ChemRegion() +
                  page.get_MusicRegion() +
                  page.get_AdvertRegion() +
                  page.get_NoiseRegion() +
                  page.get_UnknownRegion() +
                  page.get_CustomRegion())
        if oplevel == 'page' and overwrite_separators:
            page.set_SeparatorRegion([])
        else:
            ignore.extend(page.get_SeparatorRegion())
        # prepare reading order
        reading_order = dict()
        ro = page.get_ReadingOrder()
        if ro:
            rogroup = ro.get_OrderedGroup() or ro.get_UnorderedGroup()
            if rogroup:
                page_get_reading_order(reading_order, rogroup)

        # get segments to process / overwrite
        if oplevel == 'page':
            ignore.extend(page.get_TableRegion())
            regions = list(page.get_TextRegion())
            if regions:
                # page is already region-segmented
                if overwrite_regions:
                    self.logger.info(f'Removing existing TextRegions in page "{page_id}"', )
                    # we could remove all other region types as well,
                    # but this is more flexible (for workflows with
                    # specialized separator/image/table detectors):
                    page.set_TextRegion([])
                    page.set_ReadingOrder(None)
                    ro = None
                else:
                    self.logger.warning(f'Keeping existing TextRegions in page "{page_id}"', )
                    ignore.extend(regions)
            # create reading order if necessary
            if not ro or overwrite_order:
                ro = ReadingOrderType()
                page.set_ReadingOrder(ro)
            rogroup = ro.get_OrderedGroup() or ro.get_UnorderedGroup()
            if not rogroup:
                # new top-level group
                rogroup = OrderedGroupType(id="reading-order")
                ro.set_OrderedGroup(rogroup)
            if (not rogroup.get_RegionRefIndexed() and
                    not rogroup.get_OrderedGroupIndexed() and
                    not rogroup.get_UnorderedGroupIndexed()):
                # schema forbids empty OrderedGroup
                ro.set_OrderedGroup(None)
            # go get TextRegions with TextLines (and SeparatorRegions):
            image = self._process_element(page, ignore, page_image, page_coords, zoom=zoom, rogroup=rogroup)
            if image:
                result.images.append(image)
            return result

        if oplevel == 'table':
            ignore.extend(page.get_TextRegion())
            regions = list(page.get_TableRegion())
            if not regions:
                self.logger.warning(f'Page "{page_id}" contains no table regions')
            for region in regions:
                subregions = region.get_TextRegion()
                if subregions:
                    # table is already cell-segmented
                    if overwrite_regions:
                        self.logger.info(f'Removing existing TextRegions in table "{region.id}"')
                        region.set_TextRegion([])
                        roelem = reading_order.get(region.id)
                        # replace by empty group with same index and ref
                        # (which can then take the cells as subregions)
                        reading_order[region.id] = page_subgroup_in_reading_order(self.logger, roelem)
                    else:
                        self.logger.warning(f'Skipping table "{region.id}" with existing TextRegions')
                        continue
                # TODO: also allow grayscale_normalized (try/except?)
                region_image, region_coords = self.workspace.image_from_segment(
                    region, page_image, page_coords, feature_selector='binarized')
                # ignore everything but the current table region
                subignore = regions + ignore
                subignore.remove(region)
                # create reading order group if necessary
                roelem = reading_order.get(region.id)
                if not roelem:
                    self.logger.warning(
                        f"Page '{page_id}' table region '{region.id}' is not referenced in reading order "
                        f"(no target to add cells to)")
                elif overwrite_order:
                    # replace by empty ordered group with same (index and) ref
                    # (which can then take the cells as subregions)
                    roelem = page_subgroup_in_reading_order(self.logger, roelem)
                    reading_order[region.id] = roelem
                elif isinstance(roelem, (OrderedGroupType, OrderedGroupIndexedType)):
                    self.logger.warning(
                        f"Page '{page_id}' table region '{region.id}' already has an ordered group "
                        f"(cells will be appended)")
                elif isinstance(roelem, (UnorderedGroupType, UnorderedGroupIndexedType)):
                    self.logger.warning(
                        f"Page '{page_id}' table region '{region.id}' already has an unordered group "
                        f"(cells will not be appended)")
                    roelem = None
                else:
                    # replace regionRef(Indexed) by group with same index and ref
                    # (which can then take the cells as subregions)
                    roelem = page_subgroup_in_reading_order(self.logger, roelem)
                    reading_order[region.id] = roelem
                # go get TextRegions with TextLines (and SeparatorRegions)
                image = self._process_element(
                    region, subignore, region_image, region_coords, zoom=zoom, rogroup=roelem)
                if image:
                    result.images.append(image)
        else:  # 'region'
            regions = list(page.get_TextRegion())
            # besides top-level text regions, line-segment any table cells,
            # and for tables without any cells, add a pseudo-cell
            for region in page.get_TableRegion():
                subregions = region.get_TextRegion()
                if subregions:
                    regions.extend(subregions)
                else:
                    subregion = TextRegionType(
                        id=f'{region.id}_text', Coords=region.get_Coords(), parent_object_=region)
                    region.add_TextRegion(subregion)
                    regions.append(subregion)
            if not regions:
                self.logger.warning(f'Page "{page_id}" contains no text regions')
            for region in regions:
                if region.get_TextLine():
                    if overwrite_lines:
                        self.logger.info(f'Removing existing TextLines in page "{page_id}" region "{region.id}"')
                        region.set_TextLine([])
                    else:
                        self.logger.warning(f'Keeping existing TextLines in page "{page_id}" region "{region.id}"')
                        ignore.extend(region.get_TextLine())
                # TODO: also allow grayscale_normalized (try/except?)
                region_image, region_coords = self.workspace.image_from_segment(
                    region, page_image, page_coords, feature_selector='binarized')
                # if the region images have already been clipped against their neighbours specifically,
                # then we don't need to suppress all neighbours' foreground generally here
                if 'clipped' in region_coords['features'].split(','):
                    ignore = []
                # go get TextLines
                image = self._process_element(region, ignore, region_image, region_coords, zoom=zoom)
                if image:
                    result.images.append(image)
        return result

    def _process_element(self, element, ignore, image, coords, zoom=1.0, rogroup=None) -> Optional[OcrdPageResultImage]:
        """Add PAGE layout elements by segmenting an image.

        Given a PageType, TableRegionType or TextRegionType ``element``, and
        a corresponding binarized PIL.Image object ``image`` with coordinate
        metadata ``coords``, run line segmentation with Ocropy.

        If operating on the full page (or table), then also detect horizontal
        and vertical separators, and aggregate the lines into text regions
        afterwards.

        Add the resulting sub-segments to the parent ``element``.

        If ``ignore`` is not empty, then first suppress all foreground components
        in any of those segments' coordinates during segmentation, and if also
        in full page/table mode, then combine all separators among them with the
        newly detected separators to guide region segmentation.
        """
        if not image.width or not image.height:
            self.logger.warning(f"Skipping '{element.id}' with zero size")
            return None
        element_array = pil2array(image)
        element_bin = np.array(element_array <= midrange(element_array), bool)
        sep_bin = np.zeros_like(element_bin, bool)
        ignore_labels = np.zeros_like(element_bin, int)
        for i, segment in enumerate(ignore):
            self.logger.debug(
                f'Masking foreground of {type(segment).__name__[:-4]} "{segment.id}" for "{element.id}"')
            # mark these segments (e.g. separator regions, tables, images)
            # for workflows where they have been detected already;
            # these will be:
            # - ignored during text line segmentation (but not h/v-line detection)
            # - kept and reading-ordered during region segmentation (but not seps)
            segment_polygon = coordinates_of_segment(segment, image, coords)
            # If segment_polygon lies outside of element (causing
            # negative/above-max indices), either fully or partially,
            # then this will silently ignore them. The caller does
            # not need to concern herself with this.
            sp_row = segment_polygon[:, 1]
            sp_col = segment_polygon[:, 0]
            if isinstance(segment, SeparatorRegionType):
                sep_bin[draw.polygon(sp_row, sp_col, sep_bin.shape)] = True
            ignore_labels[draw.polygon(sp_row, sp_col, ignore_labels.shape)] = i + 1 #  mapped back for RO
        if isinstance(element, PageType):
            element_name = 'page'
            fullpage = True
            report = check_page(element_bin, zoom)
            suffix = '.IMG-CLIP'
        elif isinstance(element, TableRegionType) or (
                # sole/congruent text region of a table region?
                element.id.endswith('_text') and
                isinstance(element.parent_object_, TableRegionType)):
            element_name = 'table'
            fullpage = True
            report = check_region(element_bin, zoom)
            suffix = f"{element.id}.IMG-CLIP"
        else:
            element_name = 'region'
            fullpage = False
            report = check_region(element_bin, zoom)
            suffix = f"{element.id}.IMG-CLIP"
        element_name_id = f'{element_name} "{element.id}"'
        self.logger.info(f'Computing line segmentation for {element_name_id}')
        # TODO: we should downscale if DPI is large enough to save time
        try:
            if report:
                raise Exception(report)
            line_labels, baselines, seplines, images, colseps, scale = compute_segmentation(
                # suppress separators and ignored regions for textline estimation
                # but keep them for h/v-line detection (in fullpage mode):
                element_bin, seps=(sep_bin + ignore_labels) > 0,
                zoom=zoom, fullpage=fullpage,
                spread_dist=round(self.parameter['spread'] / zoom * 300 / 72),  # in pt
                # these are ignored when not in fullpage mode:
                maxcolseps=self.parameter['maxcolseps'],
                maxseps=self.parameter['maxseps'],
                maximages=self.parameter['maximages'] if element_name != 'table' else 0,
                csminheight=self.parameter['csminheight'])
        except Exception as err:
            if isinstance(element, TextRegionType):
                self.logger.error(f'Cannot line-segment region "{element.id}": {err}')
                # as a fallback, add a single text line comprising the whole region:
                element.add_TextLine(TextLineType(id=f"{element.id}_line", Coords=element.get_Coords()))
            else:
                self.logger.error(f'Cannot line-segment {element_name_id}: {err}')
            return None

        self.logger.info(f'Found {len(np.unique(line_labels)) - 1} text lines for {element_name_id}')
        # post-process line labels
        if isinstance(element, (PageType, TableRegionType)):
            # aggregate text lines to text regions
            try:
                # pass ignored regions as "line labels with initial assignment",
                # i.e. identical line and region labels
                # to detect their reading order among the others
                # (these cannot be split or grouped together with other regions)
                line_labels = np.where(line_labels, line_labels + len(ignore), ignore_labels)
                # suppress separators/images in fg and try to use for partitioning slices
                sepmask = np.maximum(sep_bin, np.maximum(seplines > 0, images > 0))
                region_labels = lines2regions(
                    element_bin, line_labels,
                    rlabels=ignore_labels,
                    sepmask=np.maximum(sepmask, colseps),  # add bg
                    # decide horizontal vs vertical cut when gaps of similar size
                    prefer_vertical=not isinstance(element, TableRegionType),
                    gap_height=self.parameter['gap_height'],
                    gap_width=self.parameter['gap_width'],
                    scale=scale, zoom=zoom)
                self.logger.info(
                    f'Found {len(np.unique(region_labels)) - 1} text regions for {element_name_id}')
            except Exception as err:
                self.logger.error(f'Cannot region-segment {element_name_id}: {err}')
                region_labels = np.where(line_labels > len(ignore), 1 + len(ignore), line_labels)

            # prepare reading order group index
            if rogroup:
                if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
                    index = 0
                    # start counting from the largest existing index
                    for elem in (rogroup.get_RegionRefIndexed() +
                                 rogroup.get_OrderedGroupIndexed() +
                                 rogroup.get_UnorderedGroupIndexed()):
                        if elem.index >= index:
                            index = elem.index + 1
                else:
                    index = None
            # find contours around region labels (can be non-contiguous):
            region_no = 0
            for region_label in np.unique(region_labels):
                if not region_label:
                    continue  # no bg
                region_mask = region_labels == region_label
                region_line_labels = line_labels * region_mask
                region_line_labels0 = np.setdiff1d(region_line_labels, [0])
                if not np.all(region_line_labels0 > len(ignore)):
                    # existing region from `ignore` merely to be ordered
                    # (no new region, no actual text lines)
                    region_line_labels0 = np.intersect1d(region_line_labels0, ignore_labels)
                    assert len(region_line_labels0) == 1, \
                        (f'Region label "{region_label}" has both existing regions and new lines '
                         f'({str(region_line_labels0)})')
                    region = ignore[region_line_labels0[0] - 1]
                    if rogroup and region.parent_object_ is element and not isinstance(region, SeparatorRegionType):
                        index = page_add_to_reading_order(rogroup, region.id, index)
                    self.logger.debug(f'Region label "{region_label}" is for ignored region "{region.id}"')
                    continue
                # normal case: new lines inside new regions
                # remove binary-empty labels, and re-order locally
                order = morph.reading_order(region_line_labels)
                order[np.setdiff1d(region_line_labels0, element_bin * region_line_labels)] = 0
                region_line_labels = order[region_line_labels]
                # avoid horizontal gaps
                region_line_labels = hmerge_line_seeds(
                    element_bin, region_line_labels, scale, seps=np.maximum(sepmask, colseps))
                region_mask |= region_line_labels > 0
                # find contours for region (can be non-contiguous)
                regions, _ = masks2polygons(
                    self.logger, region_mask * region_label, None, element_bin, name=element_name_id,
                    min_area=6000 / zoom / zoom, simplify=ignore_labels * ~(sep_bin))
                # find contours for lines (can be non-contiguous)
                lines, _ = masks2polygons(
                    self.logger, region_line_labels, baselines, element_bin, name=f'region "{element.id}"',
                    min_area=640 / zoom / zoom)
                # create new lines in new regions (allocating by intersection)
                line_polys = [Polygon(polygon) for _, polygon, _ in lines]
                for _, region_polygon, _ in regions:
                    region_poly = prep(Polygon(region_polygon))
                    # convert back to absolute (page) coordinates:
                    region_polygon = coordinates_for_segment(region_polygon, image, coords)
                    region_polygon = polygon_for_parent(region_polygon, element)
                    if region_polygon is None:
                        self.logger.warning(f'Ignoring extant region contour for region label {region_label}')
                        continue
                    # annotate result:
                    region_no += 1
                    region_id = f"{element.id}_region%04d" % region_no
                    self.logger.debug(f'Region label {region_label} becomes ID "{region_id}"')
                    region = TextRegionType(id=region_id, Coords=CoordsType(points=points_from_polygon(region_polygon)))
                    # find out which line (contours) belong to which region (contours)
                    line_no = 0
                    for i, line_poly in enumerate(line_polys):
                        if not region_poly.intersects(line_poly):  # .contains
                            continue
                        line_label, line_polygon, line_baseline = lines[i]
                        # convert back to absolute (page) coordinates:
                        line_polygon = coordinates_for_segment(line_polygon, image, coords)
                        line_polygon = polygon_for_parent(line_polygon, region)
                        if line_polygon is None:
                            self.logger.warning(
                                f'Ignoring extant line contour for region label {region_label} line label {line_label}')
                            continue
                        # annotate result:
                        line_no += 1
                        line_id = f"{region_id}_line%04d" % line_no
                        self.logger.debug(f'Line label {line_label} becomes ID "{line_id}"')
                        line = TextLineType(id=line_id, Coords=CoordsType(points=points_from_polygon(line_polygon)))
                        if line_baseline:
                            line_baseline = coordinates_for_segment(line_baseline, image, coords)
                            line.set_Baseline(BaselineType(points=points_from_polygon(line_baseline)))
                        region.add_TextLine(line)
                    # if the region has received text lines, keep it
                    if region.get_TextLine():
                        element.add_TextRegion(region)
                        self.logger.info(f'Added region "{region_id}" with {line_no} lines for {element_name_id}')
                        if rogroup:
                            index = page_add_to_reading_order(rogroup, region.id, index)
            # add additional image/non-text regions from compute_segmentation
            # (e.g. drop-capitals or images) ...
            self.logger.info(f'Found {images.max()} large image regions for {element_name_id}')
            # find contours around region labels (can be non-contiguous):
            image_polygons, _ = masks2polygons(self.logger, images, None, element_bin, name=element_name_id)
            for image_label, polygon, _ in image_polygons:
                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, image, coords)
                region_polygon = polygon_for_parent(region_polygon, element)
                if region_polygon is None:
                    self.logger.warning(f'Ignoring extant region contour for image label {image_label}')
                    continue
                region_no += 1
                # annotate result:
                region_id = f"{element.id}_image%04d" % region_no
                element.add_ImageRegion(ImageRegionType(
                    id=region_id, Coords=CoordsType(points=points_from_polygon(region_polygon))))
            # split detected separator labels into separator regions:
            self.logger.info(f'Found {seplines.max()} separators for {element_name_id}')
            # find contours around region labels (can be non-contiguous):
            sep_polygons, _ = masks2polygons(
                self.logger, seplines, None, element_bin, name=element_name_id, open_holes=True, reorder=False)
            for sep_label, polygon, _ in sep_polygons:
                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, image, coords)
                region_polygon = polygon_for_parent(region_polygon, element)
                if region_polygon is None:
                    self.logger.warning(f'Ignoring extant region contour for separator {sep_label}')
                    continue
                # annotate result:
                region_no += 1
                region_id = f"{element.id}_sep%04d" % region_no
                element.add_SeparatorRegion(SeparatorRegionType(
                    id=region_id, Coords=CoordsType(points=points_from_polygon(region_polygon))))
            # annotate a text/image-separated image
            element_array[sepmask] = np.amax(element_array)  # clip to white/bg
            image_clipped = array2pil(element_array)
            image_ref = AlternativeImageType(comments=coords['features'] + ',clipped')
            element.add_AlternativeImage(image_ref)
            return OcrdPageResultImage(image_clipped, suffix, image_ref)
        else:
            # get mask from region polygon:
            region_polygon = coordinates_of_segment(element, image, coords)
            region_mask = np.zeros_like(element_bin, bool)
            region_mask[draw.polygon(region_polygon[:, 1], region_polygon[:, 0], region_mask.shape)] = True
            # ensure the new line labels do not extrude from the region:
            line_labels = line_labels * region_mask
            # find contours around labels (can be non-contiguous):
            line_polygons, _ = masks2polygons(
                self.logger, line_labels, baselines, element_bin,
                name=f'region "{element.id}"', min_area=640 / zoom / zoom)
            line_no = 0
            for line_label, polygon, baseline in line_polygons:
                # convert back to absolute (page) coordinates:
                line_polygon = coordinates_for_segment(polygon, image, coords)
                line_polygon = polygon_for_parent(line_polygon, element)
                if line_polygon is None:
                    self.logger.warning(f'Ignoring extant line contour for line label {line_label}')
                    continue
                # annotate result:
                line_no += 1
                line_id = f"{element.id}_line%04d" % line_no
                line = TextLineType(id=line_id, Coords=CoordsType(points=points_from_polygon(line_polygon)))
                if baseline:
                    line_baseline = coordinates_for_segment(baseline, image, coords)
                    line.set_Baseline(BaselineType(points=points_from_polygon(line_baseline)))
                element.add_TextLine(line)
            if not sep_bin.any():
                return None  # no derived image
            # annotate a text/image-separated image
            element_array[sep_bin] = np.amax(element_array)  # clip to white/bg
            image_clipped = array2pil(element_array)
            image_ref = AlternativeImageType(comments=coords['features'] + ',clipped')
            element.add_AlternativeImage(image_ref)
            return OcrdPageResultImage(image_clipped, suffix, image_ref)

def polygon_for_parent(polygon, parent):
    """Clip polygon to parent polygon range.

    (Should be moved to ocrd_utils.coordinates_for_segment.)
    """
    childp = Polygon(polygon)
    if isinstance(parent, PageType):
        if parent.get_Border():
            parentp = Polygon(polygon_from_points(parent.get_Border().get_Coords().points))
        else:
            parentp = Polygon([[0,0], [0,parent.get_imageHeight()],
                               [parent.get_imageWidth(),parent.get_imageHeight()],
                               [parent.get_imageWidth(),0]])
    else:
        parentp = Polygon(polygon_from_points(parent.get_Coords().points))
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    childp = make_valid(childp)
    parentp = make_valid(parentp)
    if not childp.is_valid:
        return None
    if not parentp.is_valid:
        return None
    # check if clipping is necessary
    if childp.within(parentp):
        return childp.exterior.coords[:-1]
    # clip to parent
    interp = make_intersection(childp, parentp)
    if not interp:
        return None
    return interp.exterior.coords[:-1] # keep open

def make_intersection(poly1, poly2):
    interp = poly1.intersection(poly2)
    # post-process
    if interp.is_empty or interp.area == 0.0:
        return None
    if interp.geom_type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        interp = unary_union([geom for geom in interp.geoms if geom.area > 0])
    if interp.geom_type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        interp = join_polygons(interp.geoms)
    if interp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        interp = Polygon(np.round(interp.exterior.coords))
        interp = make_valid(interp)
    return interp

def make_valid(polygon):
    points = list(polygon.exterior.coords)
    for split in range(1, len(points)):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(points[-split:]+points[:-split])
    for tolerance in range(int(polygon.area)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance + 1)
    return polygon

def diff_polygons(poly1, poly2):
    poly = poly1.difference(poly2)
    if poly.geom_type == 'MultiPolygon':
        poly = poly.convex_hull
    if poly.minimum_clearance < 1.0:
        poly = Polygon(np.round(poly.exterior.coords))
    poly = make_valid(poly)
    return poly

def join_polygons(polygons, loc='', scale=20):
    """construct concave hull (alpha shape) from input polygons"""
    # compoundp = unary_union(polygons)
    # jointp = compoundp.convex_hull
    polygons = list(itertools.chain.from_iterable([
        poly.geoms if poly.geom_type in ['MultiPolygon', 'GeometryCollection']
        else [poly]
        for poly in polygons]))
    npoly = len(polygons)
    if npoly == 1:
        return polygons[0]
    # find min-dist path through all polygons (travelling salesman)
    pairs = itertools.combinations(range(npoly), 2)
    dists = np.eye(npoly, dtype=float)
    for i, j in pairs:
        dist = polygons[i].distance(polygons[j])
        if dist < 1e-5:
            dist = 1e-5 # if pair merely touches, we still need to get an edge
        dists[i, j] = dist
        dists[j, i] = dist
    dists = minimum_spanning_tree(dists, overwrite=True)
    # add bridge polygons (where necessary)
    max_dist = max(1.0, scale / 5)
    for prevp, nextp in zip(*dists.nonzero()):
        prevp = polygons[prevp]
        nextp = polygons[nextp]
        nearest = nearest_points(prevp, nextp)
        bridgep = LineString(nearest).buffer(max_dist, resolution=1)
        polygons.append(bridgep)
    jointp = unary_union(polygons)
    assert jointp.geom_type == 'Polygon', jointp.wkt
    if jointp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        jointp = Polygon(np.round(jointp.exterior.coords))
        jointp = make_valid(jointp)
    return jointp

def join_baselines(logger: Logger, baselines, loc=''):
    lines = []
    for baseline in baselines:
        if (baseline.is_empty or
            baseline.geom_type in ['Point', 'MultiPoint']):
            continue
        elif baseline.geom_type == 'MultiLineString':
            lines.extend(baseline.geoms)
        elif baseline.geom_type == 'LineString':
            lines.append(baseline)
        elif baseline.geom_type == 'GeometryCollection':
            for geom in baseline.geoms:
                if geom.geom_type == 'LineString':
                    lines.append(geom)
                elif geom.geom_type == 'MultiLineString':
                    lines.extend(geom)
                else:
                    logger.warning(f"Ignoring baseline subtype {geom.geom_type} in {loc}")
        else:
            logger.warning(f"Ignoring baseline type {baseline.geom_type} in {loc}")
    nlines = len(lines)
    if nlines == 0:
        return None
    elif nlines == 1:
        return lines[0]
    # Shapely cannot reorder:
    #result = line_merge(MultiLineString([line.normalize() for line in lines]))
    # find min-dist path through all lines (travelling salesman)
    pairs = itertools.combinations(range(nlines), 2)
    dists = np.eye(nlines, dtype=float)
    for i, j in pairs:
        dist = lines[i].distance(lines[j])
        if dist < 1e-5:
            dist = 1e-5 # if pair merely touches, we still need to get an edge
        dists[i, j] = dist
        dists[j, i] = dist
    dists = minimum_spanning_tree(dists, overwrite=True)
    assert dists.nonzero()[0].size, dists
    # get path
    chains = []
    for prevl, nextl in zip(*dists.nonzero()):
        foundchains = []
        for chain in chains:
            if chain[0] == prevl:
                found = chain, 0, nextl
            elif chain[0] == nextl:
                found = chain, 0, prevl
            elif chain[-1] == prevl:
                found = chain, -1, nextl
            elif chain[-1] == nextl:
                found = chain, -1, prevl
            else:
                continue
            foundchains.append(found)
        if len(foundchains):
            assert len(foundchains) <= 2, foundchains
            chain, pos, node = foundchains.pop()
            if len(foundchains):
                otherchain, otherpos, othernode = foundchains.pop()
                assert node != othernode
                assert chain[pos] == othernode
                assert otherchain[otherpos] == node
                if pos < 0 and otherpos < 0:
                    chain.extend(reversed(otherchain))
                    chains.remove(otherchain)
                elif pos < 0 and otherpos == 0:
                    chain.extend(otherchain)
                    chains.remove(otherchain)
                elif pos == 0 and otherpos == 0:
                    otherchain.extend(reversed(chain))
                    chains.remove(chain)
                elif pos == 0 and otherpos < 0:
                    otherchain.extend(chain)
                    chains.remove(chain)
            elif pos < 0:
                chain.append(node)
            else:
                chain.insert(0, node)
        else:
            chains.append([prevl, nextl])
    if len(chains) > 1:
        logger.warning(f"Baseline merge impossible (no spanning tree) in {loc}")
        return None
    assert len(chains) == 1, chains
    assert len(chains[0]) == nlines, chains[0]
    path = chains[0]
    # get points
    coords = []
    for node in path:
        line = lines[node]
        coords.extend(line.normalize().coords)
    result = LineString(coords)
    if result.is_empty:
        logger.warning(f"Baseline merge is empty in {loc}")
        return None
    assert result.geom_type == 'LineString', result.wkt
    result = set_precision(result, 1.0)
    if result.geom_type != 'LineString' or not result.is_valid:
        result = LineString(np.round(line.coords))
    return result

def page_get_reading_order(ro, rogroup):
    """Add all elements from the given reading order group to the given dictionary.

    Given a dict ``ro`` from layout element IDs to ReadingOrder element objects,
    and an object ``rogroup`` with additional ReadingOrder element objects,
    add all references to the dict, traversing the group recursively.
    """
    if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRefIndexed() +
                      rogroup.get_OrderedGroupIndexed() +
                      rogroup.get_UnorderedGroupIndexed())
    if isinstance(rogroup, (UnorderedGroupType, UnorderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRef() +
                      rogroup.get_OrderedGroup() +
                      rogroup.get_UnorderedGroup())
    for elem in regionrefs:
        ro[elem.get_regionRef()] = elem
        if not isinstance(elem, (RegionRefType, RegionRefIndexedType)):
            page_get_reading_order(ro, elem)

def page_add_to_reading_order(rogroup, region_id, index=None):
    """Add a region reference to an un/ordered RO group.

    Given a ReadingOrder group ``rogroup`` (of any type),
    append a reference to region ``region_id`` to it.

    If ``index`` is given, use that as position and return
    incremented by one. (This must be an integer if ``rogroup``
    is an OrderedGroup(Indexed).
    Otherwise return None.
    """
    if rogroup:
        if index is None:
            rogroup.add_RegionRef(RegionRefType(regionRef=region_id))
        else:
            rogroup.add_RegionRefIndexed(RegionRefIndexedType(regionRef=region_id, index=index))
            index += 1
    return index

def page_subgroup_in_reading_order(logger: Logger, roelem):
    """Replace given RO element by an equivalent OrderedGroup.

    Given a ReadingOrder element ``roelem`` (of any type),
    first look up its parent group. Remove it from the respective
    member list (of its region refs or un/ordered groups),
    even if it already was an OrderedGroup(Indexed).

    Then instantiate an empty OrderedGroup(Indexed), referencing
    the same region as ``roelem`` (and using the same index, if any).
    Add that group to the parent instead.

    Return the new group object.
    """
    if not roelem:
        logger.error('Cannot subgroup from empty ReadingOrder element')
        return roelem
    if not roelem.parent_object_:
        logger.error('Cannot subgroup from orphan ReadingOrder element')
        return roelem
    if isinstance(roelem, (OrderedGroupType, OrderedGroupIndexedType)) and not (
            roelem.get_OrderedGroupIndexed() or
            roelem.get_UnorderedGroupIndexed() or
            roelem.get_RegionRefIndexed()):
        # is already a group and still empty
        return roelem
    if isinstance(roelem, (OrderedGroupType, UnorderedGroupType, RegionRefType)):
        getattr(roelem.parent_object_, {
            OrderedGroupType: 'get_OrderedGroup',
            UnorderedGroupType: 'get_UnorderedGroup',
            RegionRefType: 'get_RegionRef',
        }.get(roelem.__class__))().remove(roelem)
        roelem2 = OrderedGroupType(id=f"{roelem.regionRef}_group", regionRef=roelem.regionRef)
        roelem.parent_object_.add_OrderedGroup(roelem2)
        roelem2.parent_object_ = roelem.parent_object_
        return roelem2
    if isinstance(roelem, (OrderedGroupIndexedType, UnorderedGroupIndexedType, RegionRefIndexedType)):
        getattr(roelem.parent_object_, {
            OrderedGroupIndexedType: 'get_OrderedGroupIndexed',
            UnorderedGroupIndexedType: 'get_UnorderedGroupIndexed',
            RegionRefIndexedType: 'get_RegionRefIndexed'
        }.get(roelem.__class__))().remove(roelem)
        roelem2 = OrderedGroupIndexedType(
            id=f"{roelem.regionRef}_group", index=roelem.index, regionRef=roelem.regionRef)
        roelem.parent_object_.add_OrderedGroupIndexed(roelem2)
        roelem2.parent_object_ = roelem.parent_object_
        return roelem2
    return None
