from __future__ import absolute_import

import os.path
import numpy as np
from skimage import draw
from skimage.morphology import convex_hull_image
import cv2
from shapely.geometry import Polygon
from shapely.prepared import prep

from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    MetadataItemType,
    LabelsType, LabelType,
    to_xml, CoordsType,
    TextLineType,
    TextRegionType,
    SeparatorRegionType,
    PageType,
    AlternativeImageType
)
from ocrd_models.ocrd_page_generateds import (
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
from ocrd_utils import (
    getLogger,
    concat_padded,
    coordinates_of_segment,
    coordinates_for_segment,
    points_from_polygon,
    polygon_from_points,
    MIMETYPE_PAGE
)

from .. import get_ocrd_tool
from .ocrolib import midrange
from .ocrolib import morph
from .common import (
    pil2array,
    array2pil,
    check_page, check_region,
    compute_segmentation,
    lines2regions
)

TOOL = 'ocrd-cis-ocropy-segment'
LOG = getLogger('processor.OcropySegment')
FALLBACK_FILEGRP_IMG = 'OCR-D-IMG-CLIP'

def masks2polygons(bg_labels, fg_bin, name, min_area=None, simplify=False):
    """Convert label masks into polygon coordinates.

    Given a Numpy array of background labels ``bg_labels``,
    and a Numpy array of the foreground ``fg_bin``,
    iterate through all labels (except zero and those labels
    which do not correspond to any foreground at all) to find
    their outer contours. Each contour part which is not too
    small and gives a (simplified) polygon of at least 4 points
    becomes a polygon. (Thus, labels can be split into multiple
    polygons.)

    Return these polygons as a list of label, polygon tuples.
    """
    results = list()
    for label in np.unique(bg_labels):
        if not label:
            # ignore if background
            continue
        bg_mask = np.array(bg_labels == label, np.uint8)
        if not np.count_nonzero(bg_mask * fg_bin):
            # ignore if missing foreground
            LOG.debug('skipping label %d in %s due to empty fg',
                      label, name)
            continue
        # simplify to convex hull
        if simplify:
            bg_mask = convex_hull_image(bg_mask).astype(np.uint8)
        # find outer contour (parts):
        contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # determine areas of parts:
        areas = [cv2.contourArea(contour) for contour in contours]
        total_area = sum(areas)
        if not total_area:
            # ignore if too small
            continue
        # sort contours in reading order
        contour_labels = np.zeros_like(bg_mask, np.uint8)
        for i, contour in enumerate(contours):
            cv2.drawContours(contour_labels, contours[i:i+1], -1, i+1, cv2.FILLED)
        order = np.argsort(morph.reading_order(contour_labels)[1:])
        # convert to polygons
        for i in order:
            contour = contours[i]
            area = areas[i]
            if min_area and area < min_area and area / total_area < 0.1:
                LOG.warning('Label %d contour %d is too small (%d/%d) in %s',
                            label, i, area, total_area, name)
                continue
            # simplify shape:
            # can produce invalid (self-intersecting) polygons:
            #polygon = cv2.approxPolyDP(contour, 2, False)[:, 0, ::] # already ordered x,y
            polygon = contour[:, 0, ::] # already ordered x,y
            # simplify and validate:
            polygon = Polygon(polygon)
            for tolerance in range(2, int(area)):
                polygon = polygon.simplify(tolerance)
                if polygon.is_valid:
                    break
            polygon = polygon.exterior.coords[:-1] # keep open
            if len(polygon) < 4:
                LOG.warning('Label %d contour %d has less than 4 points for %s',
                            label, i, name)
                continue
            results.append((label, polygon))
    return results

class OcropySegment(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools'][TOOL]
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropySegment, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            try:
                self.output_file_grp, self.image_file_grp = self.output_file_grp.split(',')
            except ValueError:
                self.image_file_grp = FALLBACK_FILEGRP_IMG
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    def process(self):
        """Segment pages into regions+lines, tables into cells+lines, or regions into lines.
        
        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the requested level.
        
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
        
        Then compute a text line segmentation for that array (as a label mask).
        When ``level-of-operation`` is ``page`` or ``table``, this also entails
        detecting
        - up to ``maximages`` large foreground images,
        - up to ``maxseps`` foreground h/v-line separators and
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

        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            file_id = input_file.ID.replace(self.input_file_grp,
                                            self.output_file_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.output_file_grp, n)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
            
            # add metadata about this operation and its runtime parameters:
            metadata = pcgts.get_Metadata() # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 Labels=[LabelsType(
                                     externalModel="ocrd-tool",
                                     externalId="parameters",
                                     Label=[LabelType(type_=name,
                                                      value=self.parameter[name])
                                            for name in self.parameter.keys()])]))
            
            # TODO: also allow grayscale_normalized (try/except?)
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id, feature_selector='binarized')
            if self.parameter['dpi'] > 0:
                zoom = 300.0/self.parameter['dpi']
            elif page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi *= 2.54
                LOG.info('Page "%s" uses %f DPI', page_id, dpi)
                zoom = 300.0/dpi
            else:
                zoom = 1

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
                        LOG.info('removing existing TextRegions in page "%s"', page_id)
                        # we could remove all other region types as well,
                        # but this is more flexible (for workflows with
                        # specialized separator/image/table detectors):
                        page.set_TextRegion([])
                        page.set_ReadingOrder(None)
                        ro = None
                    else:
                        LOG.warning('keeping existing TextRegions in page "%s"', page_id)
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
                # go get TextRegions with TextLines (and SeparatorRegions):
                self._process_element(page, ignore, page_image, page_coords,
                                      page_id, file_id, zoom, rogroup=rogroup)
            elif oplevel == 'table':
                ignore.extend(page.get_TextRegion())
                regions = list(page.get_TableRegion())
                if not regions:
                    LOG.warning('Page "%s" contains no table regions', page_id)
                for region in regions:
                    subregions = region.get_TextRegion()
                    if subregions:
                        # table is already cell-segmented
                        if overwrite_regions:
                            LOG.info('removing existing TextRegions in table "%s"', region.id)
                            region.set_TextRegion([])
                            roelem = reading_order.get(region.id)
                            # replace by empty group with same index and ref
                            # (which can then take the cells as subregions)
                            reading_order[region.id] = page_subgroup_in_reading_order(roelem)
                        else:
                            LOG.warning('skipping table "%s" with existing TextRegions', region.id)
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
                        LOG.warning("Page '%s' table region '%s' is not referenced in reading order (%s)",
                                    page_id, region.id, "no target to add cells to")
                    elif overwrite_order:
                        # replace by empty ordered group with same (index and) ref
                        # (which can then take the cells as subregions)
                        roelem = page_subgroup_in_reading_order(roelem)
                        reading_order[region.id] = roelem
                    elif isinstance(roelem, (OrderedGroupType, OrderedGroupIndexedType)):
                        LOG.warning("Page '%s' table region '%s' already has an ordered group (%s)",
                                    page_id, region.id, "cells will be appended")
                    elif isinstance(roelem, (UnorderedGroupType, UnorderedGroupIndexedType)):
                        LOG.warning("Page '%s' table region '%s' already has an unordered group (%s)",
                                    page_id, region.id, "cells will not be appended")
                        roelem = None
                    else:
                        # replace regionRef(Indexed) by group with same index and ref
                        # (which can then take the cells as subregions)
                        roelem = page_subgroup_in_reading_order(roelem)
                        reading_order[region.id] = roelem
                    # go get TextRegions with TextLines (and SeparatorRegions)
                    self._process_element(region, subignore, region_image, region_coords,
                                          region.id, file_id + '_' + region.id, zoom, rogroup=roelem)
            else: # 'region'
                regions = list(page.get_TextRegion())
                # besides top-level text regions, line-segment any table cells,
                # and for tables without any cells, add a pseudo-cell
                for region in page.get_TableRegion():
                    subregions = region.get_TextRegion()
                    if subregions:
                        regions.extend(subregions)
                    else:
                        subregion = TextRegionType(id=region.id + '_text',
                                                   Coords=region.get_Coords(),
                                                   # as if generated from parser:
                                                   parent_object_=region)
                        region.add_TextRegion(subregion)
                        regions.append(subregion)
                if not regions:
                    LOG.warning('Page "%s" contains no text regions', page_id)
                for region in regions:
                    if region.get_TextLine():
                        if overwrite_lines:
                            LOG.info('removing existing TextLines in page "%s" region "%s"', page_id, region.id)
                            region.set_TextLine([])
                        else:
                            LOG.warning('keeping existing TextLines in page "%s" region "%s"', page_id, region.id)
                            ignore.extend(region.get_TextLine())
                    # TODO: also allow grayscale_normalized (try/except?)
                    region_image, region_coords = self.workspace.image_from_segment(
                        region, page_image, page_coords, feature_selector='binarized')
                    # go get TextLines
                    self._process_element(region, ignore, region_image, region_coords,
                                          region.id, file_id + '_' + region.id, zoom)

            # update METS (add the PAGE file):
            file_path = os.path.join(self.output_file_grp, file_id + '.xml')
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

    def _process_element(self, element, ignore, image, coords, element_id, file_id, zoom=1.0, rogroup=None):
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
        element_array = pil2array(image)
        element_bin = np.array(element_array <= midrange(element_array), np.bool)
        sep_bin = np.zeros_like(element_bin, np.bool)
        ignore_labels = np.zeros_like(element_bin, np.int)
        for i, segment in enumerate(ignore):
            LOG.debug('masking foreground of %s "%s" for "%s"',
                      type(segment).__name__[:-4], segment.id, element_id)
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
            if isinstance(segment, SeparatorRegionType):
                sep_bin[draw.polygon(segment_polygon[:, 1],
                                     segment_polygon[:, 0],
                                     sep_bin.shape)] = True
            else:
                ignore_labels[draw.polygon(segment_polygon[:, 1],
                                           segment_polygon[:, 0],
                                           ignore_labels.shape)] = i+1 # mapped back for RO
        if isinstance(element, PageType):
            element_name = 'page'
            fullpage = True
            report = check_page(element_bin, zoom)
        elif isinstance(element, TableRegionType) or (
                # sole/congruent text region of a table region?
                element.id.endswith('_text') and
                isinstance(element.parent_object_, TableRegionType)):
            element_name = 'table'
            fullpage = True
            report = check_region(element_bin, zoom)
        else:
            element_name = 'region'
            fullpage = False
            report = check_region(element_bin, zoom)
        LOG.info('computing line segmentation for %s "%s"', element_name, element_id)
        # TODO: we should downscale if DPI is large enough to save time
        try:
            if report:
                raise Exception(report)
            line_labels, hlines, vlines, images, colseps, scale = compute_segmentation(
                # suppress separators and ignored regions for textline estimation
                # but keep them for h/v-line detection:
                element_bin, seps=(sep_bin+ignore_labels)>0,
                zoom=zoom, fullpage=fullpage,
                spread_dist=round(self.parameter['spread']/zoom*300/72), # in pt
                # these are ignored when not in fullpage mode:
                maxcolseps=self.parameter['maxcolseps'],
                maxseps=self.parameter['maxseps'],
                maximages=self.parameter['maximages'] if element_name != 'table' else 0,
                csminheight=self.parameter['csminheight'],
                hlminwidth=self.parameter['hlminwidth'])
        except Exception as err:
            if isinstance(element, TextRegionType):
                LOG.warning('Cannot line-segment region "%s": %s', element_id, err)
                # as a fallback, add a single text line comprising the whole region:
                element.add_TextLine(TextLineType(id=element_id + "_line", Coords=element.get_Coords()))
            else:
                LOG.error('Cannot line-segment %s "%s": %s', element_name, element_id, err)
            return

        LOG.info('Found %d text lines for %s "%s"',
                 len(np.unique(line_labels)) - 1,
                 element_name, element_id)
        # post-process line labels
        if isinstance(element, (PageType, TableRegionType)):
            # aggregate text lines to text regions
            try:
                # pass ignored regions as "line labels"
                # (which cannot be split, but may be grouped together)
                # to detect their reading order among the others
                line_labels = np.where(line_labels, line_labels+len(ignore), ignore_labels)
                # suppress separators/images in fg and try to use for partitioning slices
                sepmask = np.maximum(np.maximum(hlines, vlines),
                                     np.maximum(sep_bin, images))
                region_labels = lines2regions(
                    element_bin, line_labels,
                    sepmask=np.maximum(sepmask, colseps), # add bg
                    # decide horizontal vs vertical cut when gaps of similar size
                    prefer_vertical=not isinstance(element, TableRegionType),
                    gap_height=self.parameter['gap_height'],
                    gap_width=self.parameter['gap_width'],
                    scale=scale, zoom=zoom)
                LOG.info('Found %d text regions for %s "%s"',
                         len(np.unique(region_labels)) - 1,
                         element_name, element_id)
            except Exception as err:
                LOG.warning('Cannot region-segment %s "%s": %s',
                            element_name, element_id, err)
                region_labels = np.array(line_labels > 0, np.uint8)
            
            # prepare reading order group index
            if rogroup:
                if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
                    index = 0
                    # start counting from largest existing index
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
                    continue # no bg
                # filter text lines within this text region:
                region_line_labels = line_labels * (region_labels == region_label)
                # filter (ignored) existing regions within this region:
                region_ignore_labels = np.minimum(region_line_labels, len(ignore)+1)
                for line_label in np.argsort(morph.reading_order(region_ignore_labels)):
                    if (not line_label or
                        not np.any(element_bin * region_ignore_labels == line_label)):
                        continue
                    # split into new regions and order-only
                    if line_label <= len(ignore):
                        # existing region from `ignore` merely to be ordered
                        # (no new region, no actual text line)
                        if rogroup:
                            index = page_add_to_reading_order(rogroup, ignore[line_label-1].id, index)
                        LOG.debug('Region label %d line label %d is for ignored region "%s"',
                                  region_label, line_label, ignore[line_label-1].id)
                        continue
                    # normal case: new lines inside new regions
                    # find contours for region (can be non-contiguous)
                    region_polygons = masks2polygons(((region_labels == region_label) *
                                                      (line_labels > len(ignore))), element_bin,
                                                     '%s "%s"' % (element_name, element_id),
                                                     min_area=6000/zoom/zoom, simplify=True)
                    # find contours for lines (can be non-contiguous)
                    line_polygons = masks2polygons((region_line_labels *
                                                    (line_labels > len(ignore))), element_bin,
                                                   'region "%s"' % element_id,
                                                   min_area=640/zoom/zoom)
                    # create new lines in new regions
                    line_polys = [Polygon(polygon) for _, polygon in line_polygons]
                    for region_no, (_, region_polygon) in enumerate(region_polygons, region_no+1):
                        region_poly = prep(Polygon(region_polygon))
                        # convert back to absolute (page) coordinates:
                        region_polygon = coordinates_for_segment(region_polygon, image, coords)
                        region_polygon = polygon_for_parent(region_polygon, element)
                        # annotate result:
                        region_id = element_id + "_region%04d" % region_no
                        LOG.debug('Region label %d becomes ID "%s"', region_label, region_id)
                        region = TextRegionType(id=region_id, Coords=CoordsType(
                            points=points_from_polygon(region_polygon)))
                        # find out which line (contours) belong to which region (contours)
                        for line_no, (line_label, line_polygon) in enumerate(line_polygons):
                            line_poly = line_polys[line_no]
                            if not region_poly.intersects(line_poly): # contains
                                continue
                            # convert back to absolute (page) coordinates:
                            line_polygon = coordinates_for_segment(line_polygon, image, coords)
                            line_polygon = polygon_for_parent(line_polygon, region)
                            # annotate result:
                            line_id = region_id + "_line%04d" % line_no
                            LOG.debug('Line label %d becomes ID "%s"', line_label, line_id)
                            line = TextLineType(id=line_id, Coords=CoordsType(
                                points=points_from_polygon(line_polygon)))
                            region.add_TextLine(line)
                        if region.get_TextLine():
                            element.add_TextRegion(region)
                            LOG.info('Added region "%s" with %d lines for %s "%s"',
                                     region_id, len(line_polygons), element_name, element_id)
                            if rogroup:
                                index = page_add_to_reading_order(rogroup, region.id, index)
            # add additional image/non-text regions from compute_segmentation
            # (e.g. drop-capitals or images) ...
            image_labels, num_images = morph.label(images)
            LOG.info('Found %d large non-text/image regions for %s "%s"',
                     num_images, element_name, element_id)
            # find contours around region labels (can be non-contiguous):
            image_polygons = masks2polygons(image_labels, element_bin,
                                            '%s "%s"' % (element_name, element_id))
            for region_no, (_, polygon) in enumerate(image_polygons, region_no+1):
                region_id = element_id + "_image%04d" % region_no
                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, image, coords)
                region_polygon = polygon_for_parent(region_polygon, element)
                # annotate result:
                element.add_ImageRegion(ImageRegionType(id=region_id, Coords=CoordsType(
                    points=points_from_polygon(region_polygon))))
            # split rulers into separator regions:
            hline_labels, num_hlines = morph.label(hlines)
            vline_labels, num_vlines = morph.label(vlines)
            LOG.info('Found %d/%d h/v-lines for %s "%s"',
                     num_hlines, num_vlines, element_name, element_id)
            # find contours around region labels (can be non-contiguous):
            hline_polygons = masks2polygons(hline_labels, element_bin,
                                            '%s "%s"' % (element_name, element_id))
            vline_polygons = masks2polygons(vline_labels, element_bin,
                                            '%s "%s"' % (element_name, element_id))
            for region_no, (_, polygon) in enumerate(hline_polygons + vline_polygons, region_no+1):
                region_id = element_id + "_sep%04d" % region_no
                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, image, coords)
                region_polygon = polygon_for_parent(region_polygon, element)
                # annotate result:
                element.add_SeparatorRegion(SeparatorRegionType(id=region_id, Coords=CoordsType(
                    points=points_from_polygon(region_polygon))))
            # annotate a text/image-separated image
            element_array[sepmask] = np.amax(element_array) # clip to white/bg
            image_clipped = array2pil(element_array)
            file_path = self.workspace.save_image_file(
                image_clipped, file_id + '_clip',
                file_grp=self.image_file_grp)
            element.add_AlternativeImage(AlternativeImageType(
                filename=file_path, comments=coords['features'] + ',clipped'))
        else:
            LOG.info('Found %d text lines for region "%s"',
                     len(np.unique(line_labels)) - 1, element_id)
            # get mask from region polygon:
            region_polygon = coordinates_of_segment(element, image, coords)
            region_mask = np.zeros_like(element_bin, np.bool)
            region_mask[draw.polygon(region_polygon[:, 1],
                                     region_polygon[:, 0],
                                     region_mask.shape)] = True
            # ensure the new line labels do not extrude from the region:
            line_labels = line_labels * region_mask
            # find contours around labels (can be non-contiguous):
            line_polygons = masks2polygons(line_labels, element_bin,
                                           'region "%s"' % element_id,
                                           min_area=640/zoom/zoom)
            for line_no, (_, polygon) in enumerate(line_polygons):
                line_id = element_id + "_line%04d" % line_no
                # convert back to absolute (page) coordinates:
                line_polygon = coordinates_for_segment(polygon, image, coords)
                line_polygon = polygon_for_parent(line_polygon, element)
                # annotate result:
                element.add_TextLine(TextLineType(id=line_id, Coords=CoordsType(
                    points=points_from_polygon(line_polygon))))


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
    if childp.within(parentp):
        return polygon
    interp = childp.intersection(parentp)
    if interp.is_empty:
        # FIXME: we need a better strategy against this
        raise Exception("intersection of would-be segment with parent is empty")
    if interp.type == 'MultiPolygon':
        interp = interp.convex_hull
    return interp.exterior.coords[:-1] # keep open

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
            rogroup.add_RegionRef(RegionRefType(
                regionRef=region_id))
        else:
            rogroup.add_RegionRefIndexed(RegionRefIndexedType(
                regionRef=region_id, index=index))
            index += 1
    return index

def page_subgroup_in_reading_order(roelem):
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
        LOG.error('Cannot subgroup from empty ReadingOrder element')
        return roelem
    if not roelem.parent_object_:
        LOG.error('Cannot subgroup from orphan ReadingOrder element')
        return roelem
    if isinstance(roelem, (OrderedGroupType,OrderedGroupIndexedType)) and not (
            roelem.get_OrderedGroupIndexed() or
            roelem.get_UnorderedGroupIndexed() or
            roelem.get_RegionRefIndexed()):
        # is already a group and still empty
        return roelem
    if isinstance(roelem, (OrderedGroupType,
                           UnorderedGroupType,
                           RegionRefType)):
        getattr(roelem.parent_object_, {
            OrderedGroupType: 'get_OrderedGroup',
            UnorderedGroupType: 'get_UnorderedGroup',
            RegionRefType: 'get_RegionRef',
        }.get(roelem.__class__))().remove(roelem)
        roelem2 = OrderedGroupType(id=roelem.regionRef + '_group',
                                   regionRef=roelem.regionRef)
        roelem.parent_object_.add_OrderedGroup(roelem2)
        return roelem2
    if isinstance(roelem, (OrderedGroupIndexedType,
                           UnorderedGroupIndexedType,
                           RegionRefIndexedType)):
        getattr(roelem.parent_object_, {
            OrderedGroupIndexedType: 'get_OrderedIndexedGroup',
            UnorderedGroupIndexedType: 'get_UnorderedIndexedGroup',
            RegionRefIndexedType: 'get_RegionRefIndexed'
        }.get(roelem.__class__))().remove(roelem)
        roelem2 = OrderedGroupIndexedType(id=roelem.regionRef + '_group',
                                          index=roelem.index,
                                          regionRef=roelem.regionRef)
        roelem.parent_object_.add_OrderedGroupIndexed(roelem2)
        return roelem2
    return None
