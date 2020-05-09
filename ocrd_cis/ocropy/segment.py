from __future__ import absolute_import

import os.path
import numpy as np
from skimage import draw
import cv2
from shapely.geometry import Polygon

from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    MetadataItemType,
    LabelsType, LabelType,
    to_xml, CoordsType,
    TextLineType,
    TextRegionType,
    SeparatorRegionType,
    PageType
)
from ocrd_models.ocrd_page_generateds import (
    TableRegionType
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
    compute_segmentation,
    lines2regions
)

TOOL = 'ocrd-cis-ocropy-segment'
LOG = getLogger('processor.OcropySegment')

def masks2polygons(bg_labels, fg_bin, name):
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
            continue
        # find outer contour (parts):
        contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # determine areas of parts:
        areas = [cv2.contourArea(contour) for contour in contours]
        total_area = sum(areas)
        if not total_area:
            # ignore if too small
            continue
        for i, contour in enumerate(contours):
            area = areas[i]
            if area / total_area < 0.1:
                LOG.warning('Label %d contour %d is too small (%d/%d) in %s',
                            label, i, area, total_area, name)
                continue
            # simplify shape:
            # can produce invalid (self-intersecting) polygons:
            #polygon = cv2.approxPolyDP(contour, 2, False)[:, 0, ::] # already ordered x,y
            polygon = contour[:, 0, ::] # already ordered x,y
            polygon = Polygon(polygon).simplify(2).exterior.coords
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
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(
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

            regions = list(page.get_TextRegion())
            if oplevel == 'page':
                if regions:
                    if overwrite_regions:
                        LOG.info('removing existing TextRegions in page "%s"', page_id)
                        # we could remove all other region types as well,
                        # but this is more flexible (for workflows with
                        # specialized separator/image/table detectors):
                        page.set_TextRegion([])
                        # TODO: only remove the existing text regions from RO
                        page.set_ReadingOrder(None)
                    else:
                        LOG.warning('keeping existing TextRegions in page "%s"', page_id)
                # go get TextRegions with TextLines (and SeparatorRegions):
                self._process_element(page, page_image, page_xywh, page_id, zoom)
            else:
                for table in page.get_TableRegion():
                    subregions = table.get_TextRegion()
                    if subregions:
                        regions.extend(subregions)
                    else:
                        subregion = TextRegionType(id=table.id + '_text',
                                                   Coords=table.get_Coords(),
                                                   # as if generated from parser:
                                                   parent_object_=table)
                        table.add_TextRegion(subregion)
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
                    # TODO: also allow grayscale_normalized (try/except?)
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region, page_image, page_xywh, feature_selector='binarized')
                    # go get TextLines
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
        line segmentation with Ocropy.
        If operating on the full page (or table), aggregate lines to regions,
        and also detect horizontal and vertical separators.
        Add the resulting sub-segments to the parent ``element``.
        """
        element_array = pil2array(image)
        #element_array, _ = common.binarize(element_array, maxskew=0) # just in case still raw
        element_bin = np.array(element_array <= midrange(element_array), np.bool)
        try:
            fullpage = isinstance(element, PageType) or (
                # sole/congruent text region of a table region?
                element.id.endswith('_text') and
                isinstance(element.parent_object_, TableRegionType))
            LOG.debug('computing line segmentation for "%s" as %s',
                      element_id, 'page with columns' if fullpage else 'region')
            # TODO: we should downscale if DPI is large enough to save time
            # TODO: we should pass in existing separator masks
            # (for workflows where they have been detected/removed already)
            line_labels, hlines, vlines, colseps, scale = compute_segmentation(
                element_array,
                zoom=zoom,
                fullpage=fullpage,
                spread_dist=round(self.parameter['spread']/zoom*300/72), # in pt
                maxcolseps=self.parameter['maxcolseps'],
                maxseps=self.parameter['maxseps'],
                csminheight=self.parameter['csminheight'],
                hlminwidth=self.parameter['hlminwidth'])
        except Exception as err:
            if isinstance(element, TextRegionType):
                LOG.warning('Cannot line-segment region "%s": %s', element_id, err)
                # as a fallback, add a single text line comprising the whole region:
                element.add_TextLine(TextLineType(id=element_id + "_line", Coords=element.get_Coords()))
            else:
                LOG.error('Cannot line-segment page "%s": %s', element_id, err)
            return
        if isinstance(element, PageType):
            # aggregate text lines to text regions:
            try:
                region_labels = lines2regions(element_bin, line_labels, hlines, vlines, colseps)
            except Exception as err:
                LOG.warning('Cannot region-segment page "%s": %s', element_id, err)
                region_labels = np.array(line_labels > 0, np.uint8)
            # find contours around region labels (can be non-contiguous):
            region_polygons = masks2polygons(region_labels, element_bin, 'page "%s"' % element_id)
            for region_no, (region_label, polygon) in enumerate(region_polygons):
                region_id = element_id + "_region%04d" % region_no
                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, image, xywh)
                # annotate result:
                region = TextRegionType(id=region_id, Coords=CoordsType(
                    points=points_from_polygon(region_polygon)))
                element.add_TextRegion(region)
                # filter text lines within this text region:
                region_line_labels = line_labels * (region_labels == region_label)
                # find contours around labels (can be non-contiguous):
                line_polygons = masks2polygons(region_line_labels, element_bin, 'region "%s"' % element_id)
                for line_no, (_, polygon) in enumerate(line_polygons):
                    line_id = region_id + "_line%04d" % line_no
                    # convert back to absolute (page) coordinates:
                    line_polygon = coordinates_for_segment(polygon, image, xywh)
                    # annotate result:
                    line = TextLineType(id=line_id, Coords=CoordsType(
                        points=points_from_polygon(line_polygon)))
                    region.add_TextLine(line)
            # split rulers into separator regions:
            hline_labels, _ = morph.label(hlines)
            vline_labels, _ = morph.label(vlines)
            # find contours around region labels (can be non-contiguous):
            hline_polygons = masks2polygons(hline_labels, element_bin, 'page "%s"' % element_id)
            vline_polygons = masks2polygons(vline_labels, element_bin, 'page "%s"' % element_id)
            for region_no, (_, polygon) in enumerate(hline_polygons + vline_polygons):
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
            line_polygons = masks2polygons(line_labels, element_bin, 'region "%s"' % element_id)
            for line_no, (_, polygon) in enumerate(line_polygons):
                line_id = element_id + "_line%04d" % line_no
                # convert back to absolute (page) coordinates:
                line_polygon = coordinates_for_segment(polygon, image, xywh)
                # annotate result:
                element.add_TextLine(TextLineType(id=line_id, Coords=CoordsType(
                    points=points_from_polygon(line_polygon))))
