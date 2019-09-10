from __future__ import absolute_import

import os.path

from ocrd_utils import getLogger, concat_padded
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, AlternativeImageType, PageType
from ocrd import Processor
from ocrd_utils import MIMETYPE_PAGE

from .. import get_ocrd_tool
from . import common
from .common import (
    pil2array, array2pil
)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TOOL = 'ocrd-cis-ocropy-deskew'
LOG = getLogger('processor.OcropyDeskew')
FILEGRP_IMG = 'OCR-D-IMG-DESKEW'

def deskew(pil_image, maxskew=2):
    array = pil2array(pil_image)
    _, angle = common.binarize(array, maxskew=maxskew)
    return angle

class OcropyDeskew(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools'][TOOL]
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyDeskew, self).__init__(*args, **kwargs)

    def process(self):
        """Deskew the regions of the workspace.
        
        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the TextRegion level.
        
        Next, for each file, crop each region image according to the layout
        annotation (via coordinates into the higher-level image, or from the
        alternative image), and determine the threshold for binarization and
        the deskewing angle of the region (up to ``maxskew``). Annotate the
        angle in the region.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        level = self.parameter['level-of-operation']
        
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            file_id = input_file.ID.replace(self.input_file_grp, FILEGRP_IMG)
            if file_id == input_file.ID:
                file_id = concat_padded(FILEGRP_IMG, n)
            
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
            page_image, page_xywh, _ = self.workspace.image_from_page(
                page, page_id,
                # image must not have been rotated already,
                # (we will overwrite @orientation anyway,)
                # (This is true even if oplevel is region
                #  and page-level deskewing has been applied,
                #  because we still need to rule out rotated
                #  images on the region level, so better
                #  rotate the page level ourselves!)
                # abort if no such image can be produced:
                feature_filter='deskewed')
            if level == 'page':
                self._process_segment(page, page_image, page_xywh,
                                      "page '%s'" % page_id, input_file.pageId,
                                      file_id)
            else:
                if page_xywh['angle']:
                    LOG.info("About to rotate page '%s' by %.2f°",
                      page_id, page_xywh['angle'])
                    page_image = page_image.rotate(page_xywh['angle'],
                                                   expand=True,
                                                   #resample=Image.BILINEAR,
                                                   fillcolor='white')
                    # pretend to image_from_segment that this has *not*
                    # been rotated yet (so we can rule out images rotated
                    # on the region level):
                    #page_xywh['features'] += ',deskewed'
                    page_xywh['x'] -= round(0.5 * max(0, page_image.width  - page_xywh['w']))
                    page_xywh['y'] -= round(0.5 * max(0, page_image.height - page_xywh['h']))
                
                regions = page.get_TextRegion()
                if not regions:
                    LOG.warning('Page "%s" contains no text regions', page_id)
                for region in regions:
                    # process region:
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region, page_image, page_xywh,
                        # image must not have been rotated already,
                        # (we will overwrite @orientation anyway,)
                        # abort if no such image can be produced:
                        feature_filter='deskewed')
                    self._process_segment(region, region_image, region_xywh,
                                          "region '%s'" % region.id, input_file.pageId,
                                          file_id + '_' + region.id)
                if page_xywh['angle']:
                    # no pretense! (regardless of region results)
                    page_xywh['features'] += ',deskewed'
            
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
    
    def _process_segment(self, segment, segment_image, segment_xywh, segment_id, page_id, file_id):
        features = segment_xywh['features']
        LOG.info("About to deskew %s", segment_id)
        angle = deskew(segment_image, maxskew=self.parameter['maxskew'])
        # segment angle: PAGE orientation is defined clockwise,
        # whereas PIL/ndimage rotation is in mathematical direction:
        orientation = -angle
        orientation = 180 - (180 - orientation) % 360 # map to [-179.999,180]
        segment.set_orientation(orientation)
        LOG.info("Found angle for %s: %.1f", segment_id, angle)
        if angle:
            segment_image = segment_image.rotate(angle, expand=True,
                                                 #resample=Image.BILINEAR,
                                                 fillcolor='white')
            features += ',deskewed'
        # update METS (add the image file):
        file_path = self.workspace.save_image_file(
            segment_image,
            file_id,
            page_id=page_id,
            file_grp=FILEGRP_IMG)
        # update PAGE (reference the image file):
        segment.add_AlternativeImage(AlternativeImageType(
            filename=file_path, comments=features))
        
