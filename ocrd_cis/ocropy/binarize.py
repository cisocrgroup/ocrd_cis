from __future__ import absolute_import

import sys
import os.path
import io
import cv2
import numpy as np
from PIL import Image

#import kraken.binarization

from ocrd_utils import getLogger
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, AlternativeImageType
from ocrd import Processor
from ocrd_utils import MIMETYPE_PAGE

from .. import get_ocrd_tool
from . import common
from .common import (
    image_from_page,
    image_from_segment,
    save_image_file,
    pil2array, array2pil,
    check_line, check_page,
    # binarize,
    remove_noise)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LOG = getLogger('processor.OcropyBinarize')
FILEGRP_IMG = 'OCR-D-IMG-BIN'

def binarize(pil_image, method='ocropy', maxskew=2):
    LOG.debug('binarizing %dx%d image with method=%s', pil_image.width, pil_image.height, method)
    if method == 'none':
        return pil_image, 0
    elif method == 'ocropy':
        # parameter defaults from ocropy-nlbin:
        array = pil2array(pil_image)
        bin, angle = common.binarize(array, maxskew=maxskew)
        return array2pil(bin), angle
    # equivalent to ocropy, but without deskewing:
    # elif method == 'kraken':
    #     image = kraken.binarization.nlbin(pil_image)
    #     return image, 0
    # FIXME: add 'sauvola'
    else:
        # Convert RGB to OpenCV
        img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)

        if method == 'global':
            # global thresholding
            _, th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        elif method == 'otsu':
            # Otsu's thresholding
            _, th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif method == 'gauss-otsu':
            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            raise Exception('unknown binarization method %s', method)
        
        return Image.fromarray(th), 0


class OcropyBinarize(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-ocropy-binarize']
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyBinarize, self).__init__(*args, **kwargs)

    def process(self):
        """Binarize and deskew the pages / regions / lines of the workspace.
        
        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the requested
        `level-of-operation`.
        
        Next, for each file, crop each segment image according to the layout
        annotation (via coordinates into the higher-level image, or from the
        alternative image), and determine the threshold for binarization and
        the deskewing angle of the segment (up to `maxskew`). Apply results
        to the image and export it as an image file.
        
        Add the new image file to the workspace with a fileGrp USE equal
        `OCR-D-IMG-BIN` and an ID based on input file and input element.
        
        Reference each new image in the AlternativeImage of the element.
        
        Produce a new output file by serialising the resulting hierarchy.
        """
        method = self.parameter['method']
        maxskew = self.parameter['maxskew']
        noise_maxsize = self.parameter['noise_maxsize']
        level = self.parameter['level-of-operation']
        
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            file_id = input_file.ID.replace(self.input_file_grp, FILEGRP_IMG)
            if file_id == input_file.ID:
                file_id = concat_padded(FILEGRP_IMG, n)
            
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
            page_image, page_xywh, _ = image_from_page(
                self.workspace, page, page_id)
            
            if level == 'page':
                self.process_page(page, page_image, page_xywh,
                                  input_file.pageId, file_id)
            else:
                regions = page.get_TextRegion()
                if not regions:
                    LOG.warning('Page "%s" contains no text regions', page_id)
                for region in regions:
                    region_image, region_xywh = image_from_segment(
                        self.workspace, region, page_image, page_xywh)
                    if level == 'region':
                        self.process_region(region, region_image, region_xywh,
                                            input_file.pageId, file_id + '_' + region.id)
                        continue
                    lines = region.get_TextLine()
                    if not lines:
                        LOG.warning('Page "%s" region "%s" contains no text lines', page_id, region.id)
                    for line in lines:
                        line_image, line_xywh = image_from_segment(
                            self.workspace, line, region_image, region_xywh)
                        self.process_line(line, line_image, line_xywh,
                                          input_file.pageId, region.id,
                                          file_id + '_' + region.id + '_' + line.id)
            
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
    
    def process_page(self, page, page_image, page_xywh, page_id, file_id):
        LOG.info("About to binarize page '%s'", page_id)
        bin_image, angle = binarize(page_image,
                                    method=self.parameter['method'],
                                    maxskew=0) # FIXME self.parameter['maxskew'])
        bin_image = remove_noise(bin_image,
                                 maxsize=self.parameter['noise_maxsize'])
        # annotate angle in PAGE (to allow consumers of the AlternativeImage
        # to do consistent coordinate transforms, and non-consumers
        # to redo the rotation themselves):
        #page.set_orientation(-angle) # FIXME does not exist on page level yet!
        # update METS (add the image file):
        file_path = save_image_file(
            self.workspace,
            bin_image,
            file_id,
            page_id=page_id,
            file_grp=FILEGRP_IMG)
        # update PAGE (reference the image file):
        page.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=('grayscale_normalized' + 
                      (',cropped' if page_xywh['x'] or page_xywh['y'] else '') +
                      (',despeckled' if self.parameter['noise_maxsize'] else '') +
                      (',deskewed' if angle else ''))))
    
    def process_region(self, region, region_image, region_xywh, page_id, file_id):
        LOG.info("About to binarize page '%s' region '%s'", page_id, region.id)
        # NOTE: This just assumes that an existing TextRegion/@orientation
        # annotation is already applied in (the last) AlternativeImage if such
        # images are referenced. One could additionally check whether
        # its @comments contain the string "deskewed" (as recommended
        # by the OCR-D spec), but that would in other respects be an
        # even strong assumption.
        if region_xywh['angle']:
            # orientation has already been annotated (by previous deskewing),
            # so skip deskewing here:
            bin_image, _ = binarize(region_image,
                                    method=self.parameter['method'],
                                    maxskew=0)
        else:
            bin_image, angle = binarize(region_image,
                                        method=self.parameter['method'],
                                        maxskew=self.parameter['maxskew'])
            region_xywh['angle'] = angle
        bin_image = remove_noise(bin_image,
                                 maxsize=self.parameter['noise_maxsize'])
        # annotate angle in PAGE (to allow consumers of the AlternativeImage
        # to do consistent coordinate transforms, and non-consumers
        # to redo the rotation themselves):
        region.set_orientation(-region_xywh['angle'])
        # update METS (add the image file):
        file_path = save_image_file(
            self.workspace,
            bin_image,
            file_id=file_id,
            page_id=page_id,
            file_grp=FILEGRP_IMG)
        # update PAGE (reference the image file):
        region.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=('grayscale_normalized,cropped' + 
                      (',despeckled' if self.parameter['noise_maxsize'] else '') +
                      (',deskewed' if region_xywh['angle'] else ''))))
    
    def process_line(self, line, line_image, line_xywh, page_id, region_id, file_id):
        LOG.info("About to binarize page '%s' region '%s' line '%s'",
                 page_id, region_id, line.id)
        bin_image, angle = binarize(line_image,
                                    method=self.parameter['method'],
                                    maxskew=self.parameter['maxskew'])
        # annotate angle in PAGE (to allow consumers of the AlternativeImage
        # to do consistent coordinate transforms, and non-consumers
        # to redo the rotation themselves):
        #line.set_orientation(-angle) # does not exist on line level!
        bin_image = remove_noise(bin_image,
                                 maxsize=self.parameter['noise_maxsize'])
        # update METS (add the image file):
        file_path = save_image_file(
            self.workspace,
            bin_image,
            file_id=file_id,
            page_id=page_id,
            file_grp=FILEGRP_IMG)
        # update PAGE (reference the image file):
        line.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=('grayscale_normalized,cropped' + 
                      (',despeckled' if self.parameter['noise_maxsize'] else '') +
                      (',deskewed' if angle else ''))))
        
