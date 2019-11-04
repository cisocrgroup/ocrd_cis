from __future__ import absolute_import

import os.path
import numpy as np
from skimage import draw
import cv2

from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    to_xml, CoordsType,
    TextLineType,
    TextRegionType,
    SeparatorRegionType,
    PageType
)
from ocrd import Processor
from ocrd_utils import (
    getLogger,
    concat_padded,
    coordinates_of_segment,
    coordinates_for_segment,
    points_from_polygon,
    MIMETYPE_PAGE
)

from .. import get_ocrd_tool
from . import common
from .ocrolib import midrange
from .ocrolib import morph
from .ocrolib import sl
from .common import (
    pil2array,
    # binarize,
    compute_line_labels
)

TOOL = 'ocrd-cis-ocropy-segment'
LOG = getLogger('processor.OcropySegment')

def segment(line_labels, region_bin, region_id):
    """Convert label masks into polygon coordinates.

    Given a Numpy array of background labels ``line_labels``,
    and a Numpy array of the foreground ``region_bin``,
    iterate through all labels (except zero and those labels
    which do not correspond to any foreground at all) to find
    their outer contours. Each contour part which is not too
    small and gives a (simplified) polygon of at least 4 points
    becomes a polygon.

    Return a list of all such polygons concatenated.
    """
    lines = []
    for label in np.unique(line_labels):
        if not label:
            # ignore if background
            continue
        line_mask = np.array(line_labels == label, np.uint8)
        if not np.count_nonzero(line_mask * region_bin):
            # ignore if missing foreground
            continue
        # find outer contour (parts):
        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # determine areas of parts:
        areas = [cv2.contourArea(contour) for contour in contours]
        total_area = sum(areas)
        if not total_area:
            # ignore if too small
            continue
        for i, contour in enumerate(contours):
            area = areas[i]
            if area / total_area < 0.1:
                LOG.warning('Line label %d contour %d is too small (%d/%d) in region "%s"',
                            label, i, area, total_area, region_id)
                continue
            # simplify shape:
            polygon = cv2.approxPolyDP(contour, 2, False)[:, 0, ::] # already ordered x,y
            if len(polygon) < 4:
                LOG.warning('Line label %d contour %d has less than 4 points for region "%s"',
                            label, i, region_id)
                continue
            lines.append(polygon)
    return lines

class OcropySegment(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools'][TOOL]
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropySegment, self).__init__(*args, **kwargs)

    def process(self):
        """Segment pages into text regions or text regions into text lines.

        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the requested level.

        Next, get each element image according to the layout annotation (from
        the alternative image of the page/region, or by cropping via coordinates
        into the higher-level image), binarize it (without deskewing), and
        compute a new line segmentation for that (as a label mask).

        If ``level-of-operation`` is ``page``, aggregate text lines to text regions
        heuristically, and also detect all horizontal and up to ``maxseps`` vertical
        rulers (foreground separators), as well as up to ``maxcolseps`` column
        dividers (background separators).

        Then for each resulting segment label, convert its background mask into
        polygon outlines by finding the outer contours consistent with the element's
        polygon outline. Annotate the result by adding it as a new TextLine/TextRegion:
        If ``level-of-operation`` is ``region``, then (unless ``overwrite_lines`` is False)
        remove any existing TextLine elements, and append the new lines to the region.
        If however it is ``page``, then (unless ``overwrite_regions`` is False)
        remove any existing TextRegion elements, and append the new regions to the page.

        Produce a new output file by serialising the resulting hierarchy.
        """
        # FIXME: attempt detecting or allow passing reading order / textline order
        # FIXME: lines2regions has very coarse rules, needs bbox clustering
        # FIXME: also annotate lines already computed when level=page
        overwrite_lines = self.parameter['overwrite_lines']
        overwrite_regions = self.parameter['overwrite_regions']
        oplevel = self.parameter['level-of-operation']

        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(
                page, page_id)
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
                LOG.info('Page "%s" uses %d DPI', page_id, dpi)
                zoom = 300.0/dpi
            else:
                zoom = 1

            regions = page.get_TextRegion()
            if oplevel == 'page':
                if regions:
                    if overwrite_regions:
                        LOG.info('removing existing TextRegions in page "%s"', page_id)
                        page.set_TextRegion([])
                        page.set_ReadingOrder(None)
                    else:
                        LOG.warning('keeping existing TextRegions in page "%s"', page_id)
                self._process_element(page, page_image, page_xywh, page_id, zoom)
            else:
                if not regions:
                    LOG.warning('Page "%s" contains no text regions', page_id)
                for region in regions:
                    if region.get_TextLine():
                        if overwrite_lines:
                            LOG.info('removing existing TextLines in page "%s" region "%s"', page_id, region.id)
                            region.set_TextLine([])
                        else:
                            LOG.warning('keeping existing TextLines in page "%s" region "%s"', page_id, region.id)
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region, page_image, page_xywh)
                    self._process_element(region, region_image, region_xywh, region.id, zoom)

            # update METS (add the PAGE file):
            file_id = input_file.ID.replace(self.input_file_grp,
                                            self.output_file_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.output_file_grp, n)
            file_path = os.path.join(self.output_file_grp,
                                     file_id + '.xml')
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

    def _process_element(self, element, image, xywh, element_id, zoom):
        """Add PAGE layout elements by segmenting an image.

        Given a PageType or TextRegionType ``element``, and a corresponding
        PIL.Image object ``image`` with its bounding box ``xywh``, run
        ad-hoc binarization with Ocropy on the image (in case it was still
        raw), then a line segmentation with Ocropy. If operating on the
        full page, aggregate lines to regions. Add the resulting sub-segments
        to the parent ``element``.
        """
        # ad-hoc binarization:
        element_array = pil2array(image)
        element_array, _ = common.binarize(element_array, maxskew=0) # just in case still raw
        element_bin = np.array(element_array <= midrange(element_array), np.uint8)
        try:
            line_labels, hlines, vlines, colseps = compute_line_labels(
                element_array,
                zoom=zoom,
                fullpage=isinstance(element, PageType),
                spread_dist=round(self.parameter['spread']/zoom*300/72), # in pt
                maxcolseps=self.parameter['maxcolseps'],
                maxseps=self.parameter['maxseps'])
        except Exception as err:
            if isinstance(element, TextRegionType):
                LOG.warning('Cannot line-segment region "%s": %s', element_id, err)
                # as a fallback, add a single text line comprising the whole region:
                element.add_TextLine(TextLineType(id=element_id + "_line", Coords=element.get_Coords()))
            else:
                LOG.error('Cannot line-segment page "%s": %s', element_id, err)
            return
        #DSAVE('line labels', line_labels)
        if isinstance(element, PageType):
            # aggregate text lines to text regions:
            region_labels = self._lines2regions(line_labels, element_id)
            # find contours around region labels (can be non-contiguous):
            region_polygons = segment(region_labels, element_bin, element_id)
            for region_no, polygon in enumerate(region_polygons):
                region_id = element_id + "_region%04d" % region_no
                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, image, xywh)
                # annotate result:
                element.add_TextRegion(TextRegionType(id=region_id, Coords=CoordsType(
                    points=points_from_polygon(region_polygon))))
            # split rulers into separator regions:
            hline_labels, _ = morph.label(hlines)
            vline_labels, _ = morph.label(vlines)
            # find contours around region labels (can be non-contiguous):
            hline_polygons = segment(hline_labels, element_bin, element_id)
            vline_polygons = segment(vline_labels, element_bin, element_id)
            for region_no, polygon in enumerate(hline_polygons + vline_polygons):
                region_id = element_id + "_sep%04d" % region_no
                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, image, xywh)
                # annotate result:
                element.add_SeparatorRegion(SeparatorRegionType(id=region_id, Coords=CoordsType(
                    points=points_from_polygon(region_polygon))))
        else:
            # get mask from region polygon:
            region_polygon = coordinates_of_segment(element, image, xywh)
            region_mask = np.zeros_like(element_array)
            region_mask[draw.polygon(region_polygon[:, 1],
                                     region_polygon[:, 0],
                                     region_mask.shape)] = 1
            # ensure the new line labels do not extrude from the region:
            line_labels = line_labels * region_mask
            # find contours around labels (can be non-contiguous):
            line_polygons = segment(line_labels, element_bin, element_id)
            for line_no, polygon in enumerate(line_polygons):
                line_id = element_id + "_line%04d" % line_no
                # convert back to absolute (page) coordinates:
                line_polygon = coordinates_for_segment(polygon, image, xywh)
                # annotate result:
                element.add_TextLine(TextLineType(id=line_id, Coords=CoordsType(
                    points=points_from_polygon(line_polygon))))

    def _lines2regions(self, line_labels, page_id):
        """Aggregate text lines to text regions.

        Given a Numpy array of text lines ``line_labels``, find
        direct neighbours that match in height and are consistent
        in horizontal position. Merge these into larger region
        labels. Then morphologically close them to fill the
        background between lines. Merge regions that now contain
        each other.

        Return a Numpy array of text region labels.

        Horizontal consistency rules (in 2 passes):
        - first, aggregate pairs that flush left _and_ right
        - second, add remainders that are indented left or rugged right
        """
        objects = [None] + morph.find_objects(line_labels)
        scale = int(np.median(np.array([sl.height(obj) for obj in objects if obj])))
        num_labels = np.max(line_labels)+1
        relabel = np.arange(num_labels)
        # first determine which label pairs are adjacent:
        neighbours = np.zeros((num_labels,num_labels), np.uint8)
        for x in range(line_labels.shape[1]):
            labels = line_labels[:,x] # one column
            labels = labels[labels>0] # no background
            _, lind = np.unique(labels, return_index=True)
            labels = labels[lind] # without repetition
            neighbours[labels[:-1], labels[1:]] += 1 # successors
            neighbours[labels, labels] += 1 # identities
        # remove transitive pairs (jumping over 2 other pairs):
        for y, x in zip(*neighbours.nonzero()):
            if y == x:
                continue
            if np.any(neighbours[y, y+1:x]) and np.any(neighbours[y+1:x, x]):
                neighbours[y, x] = 0
        # now merge lines if possible (in 2 passes:
        # - first, aggregate pairs that flush left and right
        # - second, add remainders that are indented left or rugged right):
        for pass_ in ['pairs', 'remainders']:
            for label1 in range(1, num_labels - 1):
                if not neighbours[label1, label1]:
                    continue # not a label
                for label2 in range(label1 + 1, num_labels):
                    if not neighbours[label1, label2]:
                        continue # not neighbours (or transitive)
                    if relabel[label1] == relabel[label2]:
                        continue # already merged (in previous pass)
                    object1 = objects[label1]
                    object2 = objects[label2]
                    LOG.debug('page "%s" candidate lines %d (%s) vs %d (%s)',
                              page_id,
                              label1, str(sl.raster(object1)),
                              label2, str(sl.raster(object2)))
                    height = max(sl.height(object1), sl.height(object2))
                    #width = max(sl.width(object1), sl.width(object2))
                    width = line_labels.shape[1]
                    if not (
                            # similar height (e.g. discount headings from paragraph):
                            abs(sl.height(object1) - sl.height(object2)) < height * 0.2 and
                            # vertically not too far away:
                            (object2[0].start - object1[0].stop) < height * 1.0 and
                            # horizontally consistent:
                            (   # flush with each other on the left:
                                abs(object1[1].start - object2[1].start) < width * 0.1 or
                                # object1 = line indented left, object2 = block
                                (pass_ == 'remainders' and
                                 np.count_nonzero(relabel == label1) == 1 and
                                 - width * 0.1 < object1[1].start - object2[1].start < width * 0.8)
                            ) and
                            (   # flush with each other on the right:
                                abs(object1[1].stop - object2[1].stop) < width * 0.1
                                or
                                # object1 = block, object2 = line ragged right
                                (pass_ == 'pairs' and
                                 np.count_nonzero(relabel == label2) == 1 and
                                 - width * 0.1 < object1[1].stop - object2[1].stop < width * 0.8)
                            )):
                        continue
                    LOG.debug('page "%s" joining lines %d and %d', page_id, label1, label2)
                    label1 = relabel[label1]
                    label2 = relabel[label2]
                    relabel[label2] = label1
                    relabel[relabel == label2] = label1
        region_labels = relabel[line_labels]
        #DSAVE('region labels', region_labels)
        # finally, close regions:
        labels = np.unique(region_labels)
        labels = labels[labels > 0]
        for label in labels:
            region = np.array(region_labels == label)
            region_count = np.count_nonzero(region)
            if not region_count:
                continue
            LOG.debug('label %d: %d pixels', label, region_count)
            # close region between lines:
            region = morph.r_closing(region, (scale, 1))
            # extend region to background:
            region_labels = np.where(region_labels > 0, region_labels, region*label)
            # extend region to other labels that are contained in it:
            for label2 in labels[labels != label]:
                region2 = region_labels == label2
                region2_count = np.count_nonzero(region2)
                #if np.all(region2 <= region):
                # or be even more tolerant to avoid strange contours?
                if np.count_nonzero(region2 > region) < region2_count * 0.1:
                    LOG.debug('%d contains %d', label, label2)
                    region_labels[region2] = label
        #DSAVE('region labels closed', region_labels)
        return region_labels
