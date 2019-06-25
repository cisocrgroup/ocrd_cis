from __future__ import absolute_import

import sys
import os.path
import io
import warnings
import cv2
import numpy as np
from scipy.ndimage import filters, interpolation, measurements, morphology
from scipy import stats
from PIL import Image

from ocrd_utils import getLogger, concat_padded, xywh_from_points, points_from_x0y0x1y1
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, AlternativeImageType, CoordsType
from ocrd import Processor
from ocrd_utils import MIMETYPE_PAGE

from .. import get_ocrd_tool
from . import common
from .common import (
    LOG,
    image_from_page,
    image_from_region,
    image_from_line,
    save_image_file,
    pil2array, array2pil)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LOG = getLogger('processor.OcropyDeskew')

def deskew(pil_image, maxskew=2):
    array = pil2array(pil_image)
    _, angle = common.binarize(array, maxskew=maxskew)
    return angle

class OcropyDeskew(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-ocropy-binarize']
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyDeskew, self).__init__(*args, **kwargs)

    def process(self):
        """Deskew the regions of the workspace.
        
        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the TextRegion level.
        
        Next, for each file, crop each region image according to the layout
        annotation (via coordinates into the higher-level image, or from the
        alternative image), and determine the threshold for binarization and
        the deskewing angle of the region (up to `maxskew`). Annotate the
        angle in the region.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        maxskew = self.parameter['maxskew']
        
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId # (PageType has no id)
            page = pcgts.get_Page()
            page_image = self.workspace.resolve_image_as_pil(page.imageFilename)
            # process page:
            page_image, page_xywh = image_from_page(
                self.workspace, page, page_image, page_id)
            
            regions = page.get_TextRegion()
            if not regions:
                LOG.warning('Page "%s" contains no text regions', page_id)
            for region in regions:
                if region.get_orientation():
                    LOG.error('Page "%s" region "%s" already has non-zero orientation: %.1f',
                              page_id, region.id, region.get_orientation())
                    # it would be dangerous to proceed here, because
                    # the angle could already have been applied to the image
                    # and our new estimate would (or would not) be additive
                    continue
                # process region:
                region_image, region_xywh = image_from_region(
                    self.workspace, region, page_image, page_xywh, page_id)
                LOG.info("About to deskew page '%s' region '%s'",
                         page_id, region.id)
                angle = deskew(region_image, maxskew=maxskew)
                # region angle: PAGE orientation is defined clockwise,
                # whereas PIL/ndimage rotation is in mathematical direction:
                region.set_orientation(-angle)
                LOG.info("Found angle for page '%s' region '%s': %.1f",
                         page_id, region.id, angle)
            
            # update METS (add the PAGE file):
            file_id = input_file.ID.replace(self.input_file_grp,
                                            self.output_file_grp)
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
    
