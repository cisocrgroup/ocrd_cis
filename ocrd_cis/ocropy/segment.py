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
    TableRegionType,
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
    check_page, check_region,
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
            LOG.debug('skipping label %d in %s due to empty fg',
                      label, name)
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
        into the higher-level image) in binarized form, and compute a text line
        segmentation for that (as a label mask).
        
        If ``level-of-operation`` is ``page``, then afterwards aggregate text lines
        to text regions, and also detect all horizontal and up to ``maxseps`` vertical
        rulers (foreground separators), as well as up to ``maxcolseps`` column
        dividers (background separators).
        
        Then for each resulting segment label, convert its background mask into
        polygon outlines by finding the outer contours consistent with the element's
        polygon outline. Annotate the result by adding it as a new TextLine/TextRegion:
        If ``level-of-operation`` is ``region``, then (unless ``overwrite_lines`` is False)
        remove any existing TextLine elements, and append the new lines to the region.
        If however it is ``page``, then (unless ``overwrite_regions`` is False)
        remove any existing TextRegion elements, and append the new regions to the page.
        
        During line segmentation, suppress the foreground of all previously annotated
        regions (of any kind) and lines, except if just removed due to ``overwrite``.
        During region aggregation however, combine the existing separators with the
        newfound separators to guide the column search.
        
        If ``level-of-operation`` is ``page``, then afterwards iterate through all
        previously annotated (and suppressed) table regions, skipping those which
        already contain text regions (i.e. cells), but recursively entering
        page segmentation on the others (by retrieving their cropped binarized
        image etc and suppressing everything but that respective table etc).
        
        If ``level-of-operation`` is ``region``, then afterwards iterate over all
        text regions of each table region (i.e. cells) likewise. If a table does
        not contain text regions, then add a single text region covering the whole
        table and proceed.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        # FIXME: allow passing a-priori info on reading order / textline order
        overwrite_lines = self.parameter['overwrite_lines']
        overwrite_regions = self.parameter['overwrite_regions']
        overwrite_separators = self.parameter['overwrite_separators']
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
                    else:
                        LOG.warning('keeping existing TextRegions in page "%s"', page_id)
                        ignore.extend(regions)
                if not ro:
                    ro = ReadingOrderType()
                    page.set_ReadingOrder(ro)
                # create reading order if necessary
                reading_order = dict()
                ro = page.get_ReadingOrder()
                if not ro:
                    ro = ReadingOrderType()
                    page.set_ReadingOrder(ro)
                rogroup = ro.get_OrderedGroup() or ro.get_UnorderedGroup()
                if not rogroup:
                    # new top-level group
                    rogroup = OrderedGroupType(id="reading-order")
                    ro.set_OrderedGroup(rogroup)
                # go get TextRegions with TextLines (and SeparatorRegions):
                self._process_element(page, ignore, page_image, page_coords, page_id, zoom,
                                      rogroup=rogroup)
            elif oplevel == 'table':
                ignore.extend(page.get_TextRegion())
                regions = list(page.get_TableRegion())
                for region in page.get_TableRegion():
                    subregions = region.get_TextRegion()
                    if subregions:
                        # table is already cell-segmented
                        if overwrite_regions:
                            LOG.info('removing existing TextRegions in table "%s"', region.id)
                            region.set_TextRegion([])
                            roelem = reading_order.get(region.id)
                            if isinstance(roelem, (OrderedGroupType,UnorderedGroupType,RegionRefType)):
                                getattr(roelem.parent_object_, {
                                    OrderedGroupType: 'get_OrderedGroup',
                                    UnorderedGroupType: 'get_UnorderedGroup',
                                    RegionRefType: 'get_RegionRef',
                                }.get(roelem.__class__))().remove(roelem)
                                roelem2 = RegionRefType(id=region.id + '_ref',
                                                        regionRef=roelem.regionRef)
                                roelem.parent_object_.add_OrderedGroup(roelem2)
                                reading_order[region.id] = roelem2
                            elif isinstance(roelem, (OrderedGroupType,UnorderedGroupType,RegionRefType)):
                                getattr(roelem.parent_object_, {
                                    OrderedGroupIndexedType: 'get_OrderedIndexedGroup',
                                    UnorderedGroupIndexedType: 'get_UnorderedIndexedGroup',
                                    RegionRefIndexedType: 'get_RegionRefIndexed'
                                }.get(roelem.__class__))().remove(roelem)
                                roelem2 = RegionRefIndexedType(id=region.id + '_ref',
                                                               index=roelem.index,
                                                               regionRef=roelem.regionRef)
                                roelem.parent_object_.add_OrderedGroupIndexed(roelem2)
                                reading_order[region.id] = roelem2
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
                    elif isinstance(roelem, (OrderedGroupType, OrderedGroupIndexedType)):
                        LOG.warning("Page '%s' table region '%s' already has an ordered group (%s)",
                                    page_id, region.id, "cells will be appended")
                    elif isinstance(roelem, (UnorderedGroupType, UnorderedGroupIndexedType)):
                        LOG.warning("Page '%s' table region '%s' already has an unordered group (%s)",
                                    page_id, region.id, "cells will not be appended")
                        roelem = None
                    elif isinstance(roelem, RegionRefIndexedType):
                        # replace regionref by group with same index and ref
                        # (which can then take the cells as subregions)
                        roelem2 = OrderedGroupIndexedType(id=region.id + '_order',
                                                          index=roelem.index,
                                                          regionRef=roelem.regionRef)
                        roelem.parent_object_.add_OrderedGroupIndexed(roelem2)
                        roelem.parent_object_.get_RegionRefIndexed().remove(roelem)
                        roelem = roelem2
                    elif isinstance(roelem, RegionRefType):
                        # replace regionref by group with same ref
                        # (which can then take the cells as subregions)
                        roelem2 = OrderedGroupType(id=region.id + '_order',
                                                   regionRef=roelem.regionRef)
                        roelem.parent_object_.add_OrderedGroup(roelem2)
                        roelem.parent_object_.get_RegionRef().remove(roelem)
                        roelem = roelem2
                    # go get TextRegions with TextLines (and SeparatorRegions)
                    self._process_element(region, subignore, region_image, region_coords, region.id, zoom,
                                          rogroup=roelem)
            else:
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
                    self._process_element(region, ignore, region_image, region_coords, region.id, zoom)

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

    def _process_element(self, element, ignore, image, coords, element_id, zoom=1.0, rogroup=None):
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
        #element_array, _ = common.binarize(element_array, maxskew=0) # just in case still raw
        element_bin = np.array(element_array <= midrange(element_array), np.bool)
        sep_bin = np.zeros_like(element_bin, np.bool)
        for segment in ignore:
            LOG.debug('suppressing foreground of %s "%s" for "%s"',
                      type(segment).__name__[:-4], segment.id, element_id)
            # suppress these segments' foreground (e.g. separator regions)
            # (for workflows where they have been detected already)
            segment_polygon = coordinates_of_segment(segment, image, coords)
            # If segment_polygon lies outside of element (causing
            # negative/above-max indices), either fully or partially,
            # then this will silently ignore them. The caller does
            # not need to concern herself with this.
            element_bin[draw.polygon(segment_polygon[:, 1],
                                     segment_polygon[:, 0],
                                     element_bin.shape)] = False
            if isinstance(segment, SeparatorRegionType):
                sep_bin[draw.polygon(segment_polygon[:, 1],
                                     segment_polygon[:, 0],
                                     sep_bin.shape)] = True
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
            line_labels, hlines, vlines, colseps, scale = compute_segmentation(
                element_bin, seps=sep_bin, zoom=zoom, fullpage=fullpage,
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
                LOG.error('Cannot line-segment %s "%s": %s', element_name, element_id, err)
            return

        # post-process line labels
        if isinstance(element, (PageType, TableRegionType)):
            # aggregate text lines to text regions:
            try:
                region_labels = lines2regions(element_bin, line_labels, hlines, vlines,
                                              np.maximum(colseps, sep_bin))
                LOG.info('Found %d text lines in %d text regions for %s "%s"',
                         len(np.unique(line_labels)) - 1,
                         len(np.unique(region_labels)) - 1,
                         element_name, element_id)
            except Exception as err:
                LOG.warning('Cannot region-segment %s "%s": %s',
                            element_name, element_id, err)
                region_labels = np.array(line_labels > 0, np.uint8)
            # find contours around region labels (can be non-contiguous):
            region_polygons = masks2polygons(region_labels, element_bin,
                                             '%s "%s"' % (element_name, element_id))
            for region_no, (region_label, polygon) in enumerate(region_polygons):
                region_id = element_id + "_region%04d" % region_no
                LOG.debug('Region label %d becomes ID "%s"', region_label, region_id)
                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, image, coords)
                # annotate result:
                region = TextRegionType(id=region_id, Coords=CoordsType(
                    points=points_from_polygon(region_polygon)))
                element.add_TextRegion(region)
                # filter text lines within this text region:
                region_line_labels = line_labels * (region_labels == region_label)
                # find contours around labels (can be non-contiguous):
                line_polygons = masks2polygons(region_line_labels, element_bin,
                                               'region "%s"' % region_id)
                for line_no, (line_label, polygon) in enumerate(line_polygons):
                    line_id = region_id + "_line%04d" % line_no
                    LOG.debug('Line label %d becomes ID "%s"', line_label, line_id)
                    # convert back to absolute (page) coordinates:
                    line_polygon = coordinates_for_segment(polygon, image, coords)
                    # annotate result:
                    line = TextLineType(id=line_id, Coords=CoordsType(
                        points=points_from_polygon(line_polygon)))
                    region.add_TextLine(line)
                LOG.info('Added region "%s" with %d lines for %s "%s"',
                         region_id, len(line_polygons), element_name, element_id)
                # add to reading order
                if rogroup:
                    if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
                        index = 0
                        # start counting from largest existing index
                        for elem in (rogroup.get_RegionRefIndexed() +
                                     rogroup.get_OrderedGroupIndexed() +
                                     rogroup.get_UnorderedGroupIndexed()):
                            if elem.index >= index:
                                index = elem.index + 1
                        rogroup.add_RegionRefIndexed(RegionRefIndexedType(
                            regionRef=region.id, index=index))
                    else:
                        rogroup.add_RegionRef(RegionRefType(
                            regionRef=region.id))
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
            for region_no, (_, polygon) in enumerate(hline_polygons + vline_polygons):
                region_id = element_id + "_sep%04d" % region_no
                # convert back to absolute (page) coordinates:
                region_polygon = coordinates_for_segment(polygon, image, coords)
                # annotate result:
                element.add_SeparatorRegion(SeparatorRegionType(id=region_id, Coords=CoordsType(
                    points=points_from_polygon(region_polygon))))
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
                                           'region "%s"' % element_id)
            for line_no, (_, polygon) in enumerate(line_polygons):
                line_id = element_id + "_line%04d" % line_no
                # convert back to absolute (page) coordinates:
                line_polygon = coordinates_for_segment(polygon, image, coords)
                # annotate result:
                element.add_TextLine(TextLineType(id=line_id, Coords=CoordsType(
                    points=points_from_polygon(line_polygon))))

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
