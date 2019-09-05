from __future__ import absolute_import

import sys
import os.path
import numpy as np
from skimage import draw
import cv2

from ocrd_utils import getLogger
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, CoordsType, TextLineType
from ocrd import Processor
from ocrd_utils import (
    coordinates_of_segment,
    coordinates_for_segment,
    points_from_polygon,
    MIMETYPE_PAGE
)

from .. import get_ocrd_tool
from . import common
from .ocrolib import midrange
from .common import (
    pil2array, array2pil,
    check_line, check_page,
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
        for i in range(len(contours)):
            contour = contours[i]
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
        """Segment pages or regions of the workspace into lines.
        
        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the requested level.
        
        Next, get each segment image according to the layout annotation (from
        the alternative image of the page/region, or by cropping via coordinates
        into the higher-level image), binarize it (without deskewing), and
        compute a new line segmentation for that (as a label mask).
        
        Then for each line label, convert its background mask into polygon outlines
        by finding the outer contours consistent with the segment's polygon outline.
        Annotate the result as a new TextLine element: If ``level-of-operation``
        is ``region``, then (unless ``overwrite_lines`` is False) remove any existing
        TextLine elements, and append the new lines to the region. If however it
        is ``page``, then (unless ``overwrite_regions`` is False) remove any existing
        TextRegion elements, and aggregate the lines into useful regions, giving extra
        consideration to columns and separators.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        overwrite_lines = self.parameter['overwrite_lines']
        # FIXME: add level-of-operation page
        # FIXME: expose some parameters
        
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
            if not regions:
                LOG.warning('Page "%s" contains no text regions', page_id)
            for region in regions:
                if region.get_TextLine():
                    if overwrite_lines:
                        LOG.info('removing existing TextLines in region "%s"', region.id)
                        region.set_TextLine([])
                    else:
                        LOG.warning('keeping existing TextLines in region "%s"', region.id)
                region_image, region_xywh = self.workspace.image_from_segment(
                    region, page_image, page_xywh)
                # ad-hoc binarization:
                region_array = pil2array(region_image)
                region_array, _ = common.binarize(region_array, maxskew=0) # just in case still raw
                region_bin = np.array(region_array <= midrange(region_array), np.uint8)
                try:
                    line_labels = compute_line_labels(region_array, zoom=zoom, fullpage=False)
                except Exception as err:
                    LOG.warning('Cannot line-segment page "%s" region "%s": %s',
                                page_id, region.id, err)
                    region.add_TextLine(TextLineType(id=region.id + "_line", Coords=region.get_Coords()))
                    continue
                # mask from region polygon:
                region_polygon = coordinates_of_segment(region, region_image, region_xywh)
                region_mask = np.zeros_like(region_array)
                region_mask[draw.polygon(region_polygon[:,1], region_polygon[:,0], region_mask.shape)] = 1
                line_labels = line_labels * region_mask
                # find contours around labels (can be non-contiguous):
                line_polygons = segment(line_labels, region_bin, region.id)
                for line_no, polygon in enumerate(line_polygons):
                    line_id = region.id + "_line%04d" % line_no
                    # convert back to absolute (page) coordinates:
                    line_polygon = coordinates_for_segment(polygon, region_image, region_xywh)
                    # annotate result:
                    region.add_TextLine(TextLineType(id=line_id, Coords=CoordsType(
                        points=points_from_polygon(line_polygon))))
            
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
    
