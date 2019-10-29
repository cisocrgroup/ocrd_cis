from __future__ import absolute_import

import os.path
import cv2
import numpy as np
from PIL import Image

#import kraken.binarization

from ocrd_utils import (
    getLogger,
    concat_padded,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, AlternativeImageType
from ocrd import Processor

from .. import get_ocrd_tool
from . import common
from .common import (
    pil2array, array2pil,
    # binarize,
    remove_noise)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LOG = getLogger('processor.OcropyBinarize')
FALLBACK_FILEGRP_IMG = 'OCR-D-IMG-BIN'

def binarize(pil_image, method='ocropy', maxskew=2, nrm=False):
    LOG.debug('binarizing %dx%d image with method=%s', pil_image.width, pil_image.height, method)
    if method == 'none':
        return pil_image, 0
    elif method == 'ocropy':
        # parameter defaults from ocropy-nlbin:
        array = pil2array(pil_image)
        bin, angle = common.binarize(array, maxskew=maxskew, nrm=nrm)
        return array2pil(bin), angle
    # equivalent to ocropy, but without deskewing:
    # elif method == 'kraken':
    #     image = kraken.binarization.nlbin(pil_image)
    #     return image, 0
    # FIXME: add 'sauvola'
    else:
        # Convert RGB to OpenCV
        #img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)
        img = np.asarray(pil_image.convert('L'))

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
            raise Exception('unknown binarization method %s' % method)
        return Image.fromarray(th), 0


class OcropyBinarize(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-ocropy-binarize']
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyBinarize, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            if self.parameter['grayscale'] and self.parameter['method'] != 'ocropy':
                LOG.critical('requested method %s does not support grayscale normalized output',
                             self.parameter['method'])
                raise Exception('only method=ocropy allows grayscale=true')
            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    def process(self):
        """Binarize (and optionally deskew/despeckle) the pages/regions/lines of the workspace.

        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the requested
        ``level-of-operation``.

        Next, for each file, crop each segment image according to the layout
        annotation (via coordinates into the higher-level image, or from the
        alternative image), and determine the threshold for binarization and
        the deskewing angle of the segment (up to ``maxskew``). Then despeckle
        by removing connected components smaller than ``noise_maxsize``.
        Finally, apply results to the image and export it as an image file.

        Add the new image file to the workspace with the fileGrp USE given
        in the second position of the output fileGrp, or ``OCR-D-IMG-BIN``,
        and an ID based on input file and input element.

        Reference each new image in the AlternativeImage of the element.

        Produce a new output file by serialising the resulting hierarchy.
        """
        level = self.parameter['level-of-operation']

        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.image_grp, n)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
            page_image, page_xywh, _ = self.workspace.image_from_page(
                page, page_id, feature_filter='binarized')
            
            if level == 'page':
                self.process_page(page, page_image, page_xywh,
                                  input_file.pageId, file_id)
            else:
                regions = page.get_TextRegion() + (
                    page.get_TableRegion() if level == 'region' else [])
                if not regions:
                    LOG.warning('Page "%s" contains no text regions', page_id)
                for region in regions:
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region, page_image, page_xywh, feature_filter='binarized')
                    if level == 'region':
                        self.process_region(region, region_image, region_xywh,
                                            input_file.pageId, file_id + '_' + region.id)
                        continue
                    lines = region.get_TextLine()
                    if not lines:
                        LOG.warning('Page "%s" region "%s" contains no text lines', page_id, region.id)
                    for line in lines:
                        line_image, line_xywh = self.workspace.image_from_segment(
                            line, region_image, region_xywh, feature_filter='binarized')
                        self.process_line(line, line_image, line_xywh,
                                          input_file.pageId, region.id,
                                          file_id + '_' + region.id + '_' + line.id)

            # update METS (add the PAGE file):
            file_id = input_file.ID.replace(self.input_file_grp, self.page_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.page_grp, n)
            file_path = os.path.join(self.page_grp, file_id + '.xml')
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.page_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.page_grp, out.local_filename)

    def process_page(self, page, page_image, page_xywh, page_id, file_id):
        LOG.info("About to binarize page '%s'", page_id)
        features = page_xywh['features']
        if 'angle' in page_xywh and page_xywh['angle']:
            # orientation has already been annotated (by previous deskewing),
            # so skip deskewing here:
            bin_image, _ = binarize(page_image,
                                    method=self.parameter['method'],
                                    maxskew=0,
                                    nrm=self.parameter['grayscale'])
        else:
            bin_image, angle = binarize(page_image,
                                        method=self.parameter['method'],
                                        maxskew=self.parameter['maxskew'],
                                        nrm=self.parameter['grayscale'])
            if angle:
                features += ',deskewed'
            page_xywh['angle'] = angle
        bin_image = remove_noise(bin_image,
                                 maxsize=self.parameter['noise_maxsize'])
        if self.parameter['noise_maxsize']:
            features += ',despeckled'
        # annotate angle in PAGE (to allow consumers of the AlternativeImage
        # to do consistent coordinate transforms, and non-consumers
        # to redo the rotation themselves):
        orientation = -page_xywh['angle']
        orientation = 180 - (180 - orientation) % 360 # map to [-179.999,180]
        page.set_orientation(orientation)
        # update METS (add the image file):
        if self.parameter['grayscale']:
            file_id += '.nrm'
            features += ',grayscale_normalized'
        else:
            features += ',binarized'
        file_path = self.workspace.save_image_file(
            bin_image,
            file_id,
            page_id=page_id,
            file_grp=self.image_grp)
        # update PAGE (reference the image file):
        page.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=features))

    def process_region(self, region, region_image, region_xywh, page_id, file_id):
        LOG.info("About to binarize page '%s' region '%s'", page_id, region.id)
        features = region_xywh['features']
        if 'angle' in region_xywh and region_xywh['angle']:
            # orientation has already been annotated (by previous deskewing),
            # so skip deskewing here:
            bin_image, _ = binarize(region_image,
                                    method=self.parameter['method'],
                                    maxskew=0,
                                    nrm=self.parameter['grayscale'])
        else:
            bin_image, angle = binarize(region_image,
                                        method=self.parameter['method'],
                                        maxskew=self.parameter['maxskew'],
                                        nrm=self.parameter['grayscale'])
            if angle:
                features += ',deskewed'
            region_xywh['angle'] = angle
        bin_image = remove_noise(bin_image,
                                 maxsize=self.parameter['noise_maxsize'])
        if self.parameter['noise_maxsize']:
            features += ',despeckled'
        # annotate angle in PAGE (to allow consumers of the AlternativeImage
        # to do consistent coordinate transforms, and non-consumers
        # to redo the rotation themselves):
        orientation = -region_xywh['angle']
        orientation = 180 - (180 - orientation) % 360 # map to [-179.999,180]
        region.set_orientation(orientation)
        # update METS (add the image file):
        if self.parameter['grayscale']:
            file_id += '.nrm'
            features += ',grayscale_normalized'
        else:
            features += ',binarized'
        file_path = self.workspace.save_image_file(
            bin_image,
            file_id,
            page_id=page_id,
            file_grp=self.image_grp)
        # update PAGE (reference the image file):
        region.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=features))

    def process_line(self, line, line_image, line_xywh, page_id, region_id, file_id):
        LOG.info("About to binarize page '%s' region '%s' line '%s'",
                 page_id, region_id, line.id)
        features = line_xywh['features']
        bin_image, angle = binarize(line_image,
                                    method=self.parameter['method'],
                                    maxskew=self.parameter['maxskew'],
                                    nrm=self.parameter['grayscale'])
        if angle:
            features += ',deskewed'
        # annotate angle in PAGE (to allow consumers of the AlternativeImage
        # to do consistent coordinate transforms, and non-consumers
        # to redo the rotation themselves):
        #orientation = -angle
        #orientation = 180 - (180 - orientation) % 360 # map to [-179.999,180]
        #line.set_orientation(orientation) # does not exist on line level!
        LOG.warning("cannot add orientation %.2f to page '%s' region '%s' line '%s'",
                    -angle, page_id, region_id, line.id)
        bin_image = remove_noise(bin_image,
                                 maxsize=self.parameter['noise_maxsize'])
        if self.parameter['noise_maxsize']:
            features += ',despeckled'
        # update METS (add the image file):
        if self.parameter['grayscale']:
            file_id += '.nrm'
            features += ',grayscale_normalized'
        else:
            features += ',binarized'
        file_path = self.workspace.save_image_file(
            bin_image,
            file_id,
            page_id=page_id,
            file_grp=self.image_grp)
        # update PAGE (reference the image file):
        line.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=features))
