from __future__ import absolute_import

import sys
import os.path
import io
import numpy as np
from PIL import Image, ImageStat
from scipy.ndimage import filters

from ocrd_utils import getLogger
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    to_xml, AlternativeImageType,
    TextRegionType, TextLineType
)
from ocrd_models import OcrdExif
from ocrd import Processor
from ocrd_utils import MIMETYPE_PAGE

from .. import get_ocrd_tool
from . import common
from .ocrolib import midrange, morph
from .common import (
    coordinates_of_segment,
    xywh_from_points,
    bbox_from_polygon,
    image_from_page,
    image_from_region,
    image_from_polygon,
    polygon_mask,
    crop_image,
    pil2array, array2pil,
    # binarize,
    save_image_file
)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TOOL = 'ocrd-cis-ocropy-clip'
LOG = getLogger('processor.OcropyClip')
FILEGRP_IMG = 'OCR-D-IMG-CLIP'

class OcropyClip(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools'][TOOL]
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyClip, self).__init__(*args, **kwargs)

    def process(self):
        """Clip text regions / lines of the workspace at intersections with neighbours.
        
        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the requested
        `level-of-operation`.
        
        Next, get each segment image according to the layout annotation (by cropping
        via coordinates into the higher-level image), as well as all its neighbours',
        binarize them (without deskewing), and make a connected component analysis.
        (Segments must not already have AlternativeImage or orientation angle
         annotated, otherwise they will be skipped.)
        
        Then, for each section of overlap with a neighbour, re-assign components
        which are only contained in the neighbour by clipping them to white (background),
        and export the (final) result as image file.
        
        Add the new image file to the workspace with a fileGrp USE equal
        `OCR-D-IMG-CLIP` and an ID based on the input file and input element.
        
        Reference each new image in the AlternativeImage of the element.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        # This makes best sense for overlapping segmentation, like current GT
        # or Tesseract layout analysis. Most notably, it can suppress graphics
        # and separators within or across a region or line. It _should_ ideally
        # be run after binarization (on page level for region-level clipping,
        # and on the region level for line-level clipping), because the
        # connected component analysis after implicit binarization could be
        # suboptimal, and the explicit binarization after clipping could be,
        # too. However, region-level clipping _must_ be run before region-level
        # deskewing, because that would make segments incomensurable with their
        # neighbours.
        level = self.parameter['level-of-operation']
        
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            file_id = input_file.ID.replace(self.input_file_grp, FILEGRP_IMG)
            if file_id == input_file.ID:
                file_id = concat_padded(FILEGRP_IMG, n)
            
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
            page_image = self.workspace.resolve_image_as_pil(page.imageFilename)
            page_image_info = OcrdExif(page_image)
            if page_image_info.xResolution != 1:
                dpi = page_image_info.xResolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
                LOG.info('Page "%s" uses %d DPI', page_id, dpi)
                zoom = 300.0/dpi
            else:
                zoom = 1
            page_image, page_xywh = image_from_page(
                self.workspace, page, page_image, page_id)
            
            regions = page.get_TextRegion()
            other_regions = (
                page.get_AdvertRegion() + 
                page.get_ChartRegion() +
                page.get_ChemRegion() +
                page.get_GraphicRegion() +
                page.get_ImageRegion() +
                page.get_LineDrawingRegion() +
                page.get_MathsRegion() +
                page.get_MusicRegion() +
                page.get_NoiseRegion() +
                page.get_SeparatorRegion() +
                page.get_TableRegion() +
                page.get_UnknownRegion())
            if not regions:
                LOG.warning('Page "%s" contains no text regions', page_id)
            for i, region in enumerate(regions):
                if level == 'region':
                    if region.get_AlternativeImage():
                        LOG.warning('Page "%s" region "%s" already contains image data: skipping',
                                    page_id, region.id)
                        continue
                    if region.get_orientation():
                        LOG.warning('Page "%s" region "%s" has non-zero orientation: skipping',
                                    page_id, region.id)
                        continue
                    self.process_segment(region, regions[:i] + regions[i+1:] + other_regions,
                                         page_image, page_xywh,
                                         input_file.pageId, file_id + '_' + region.id)
                    continue
                region_image, region_xywh = image_from_region(
                    self.workspace, region, page_image, page_xywh)
                lines = region.get_TextLine()
                if not lines:
                    LOG.warning('Page "%s" region "%s" contains no text lines', page_id, region.id)
                    continue
                for j, line in enumerate(lines):
                    if line.get_AlternativeImage():
                        LOG.warning('Page "%s" region "%s" line "%s" already contains image data: skipping',
                                    page_id, region.id, line.id)
                        continue
                    self.process_segment(line, lines[:j] + lines[j+1:],
                                         region_image, region_xywh,
                                         input_file.pageId, file_id + '_' + region.id + '_' + line.id)
            
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
    
    def process_segment(self, segment, neighbours, parent_image, parent_xywh, page_id, file_id):
        segment_xywh = xywh_from_points(segment.get_Coords().points)
        segment_polygon = coordinates_of_segment(segment, parent_image, parent_xywh)
        segment_bbox = bbox_from_polygon(segment_polygon)
        segment_image = image_from_polygon(parent_image, segment_polygon)
        background = ImageStat.Stat(segment_image).median[0]
        background_image = Image.new('L', segment_image.size, background)
        segment_mask = pil2array(polygon_mask(parent_image, segment_polygon)).astype(np.uint8)
        # ad-hoc binarization:
        parent_array = pil2array(parent_image)
        parent_array, _ = common.binarize(parent_array, maxskew=0) # just in case still raw
        parent_bin = np.array(parent_array <= midrange(parent_array), np.uint8)
        for neighbour in neighbours:
            neighbour_polygon = coordinates_of_segment(neighbour, parent_image, parent_xywh)
            neighbour_bbox = bbox_from_polygon(neighbour_polygon)
            # not as precise as a (mutual) polygon intersection test, but that would add
            # a dependency on `shapely` (and we only loose a little speed here):
            if not (segment_bbox[2] >= neighbour_bbox[0] and
                    neighbour_bbox[2] >= segment_bbox[0] and
                    segment_bbox[3] >= neighbour_bbox[1] and
                    neighbour_bbox[3] >= segment_bbox[1]):
                continue
            neighbour_mask = pil2array(polygon_mask(parent_image, neighbour_polygon)).astype(np.uint8)
            # extend mask by 3 pixel in each direction to ensure it does not leak components accidentally
            # (accounts for bad cropping of non-text regions in GT):
            if not isinstance(neighbour, (TextRegionType, TextLineType)):
                neighbour_mask = filters.maximum_filter(neighbour_mask, 7)
            # find connected components that (only) belong to the neighbour:
            intruders = segment_mask * morph.keep_marked(parent_bin, neighbour_mask > 0) # overlaps neighbour
            intruders -= morph.keep_marked(intruders, segment_mask - neighbour_mask > 0) # but exclusively
            num_intruders = np.count_nonzero(intruders)
            num_foreground = np.count_nonzero(segment_mask * parent_bin)
            if not num_intruders:
                continue
            if num_intruders / num_foreground > 1.0 - self.parameter['min_fraction']:
                LOG.info('Too many intruders (%d/%d) from neighbour "%s" in segment "%s" on page "%s"',
                         num_intruders, num_foreground, neighbour.id, segment.id, page_id)
                continue
            LOG.debug('segment "%s" vs neighbour "%s": suppressing %d pixels on page "%s"',
                      segment.id, neighbour.id, np.count_nonzero(intruders), page_id)
            clip_mask = array2pil(intruders)
            #parent_bin[intruders] = 0 # suppress in binary for next iteration
            segment_image.paste(background_image, mask=clip_mask) # suppress in raw image
        # recrop segment into rectangle (also clipping with white):
        segment_image = crop_image(segment_image,
            box=(segment_xywh['x'] - parent_xywh['x'],
                 segment_xywh['y'] - parent_xywh['y'],
                 segment_xywh['x'] - parent_xywh['x'] + segment_xywh['w'],
                 segment_xywh['y'] - parent_xywh['y'] + segment_xywh['h']))
        # update METS (add the image file):
        file_path = save_image_file(
            self.workspace,
            segment_image,
            file_id=file_id,
            page_id=page_id,
            file_grp=FILEGRP_IMG)
        # update PAGE (reference the image file):
        segment.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments='cropped,clipped'))
