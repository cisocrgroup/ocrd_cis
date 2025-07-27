from __future__ import absolute_import
from logging import Logger
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ocrd_utils import getLogger
from ocrd_models.ocrd_page import AlternativeImageType, OcrdPage
from ocrd import Processor, OcrdPageResult, OcrdPageResultImage

from . import common
from .common import array2pil, determine_zoom, pil2array, remove_noise


def binarize(logger: Logger, pil_image, method='ocropy', maxskew=2, threshold=0.5, nrm=False, zoom=1.0):
    logger.debug(f'Binarizing {pil_image.width}x{pil_image.height} image with method={method}')
    if method == 'none':
        # useful if the images are already binary,
        # but lack image attribute `binarized`
        return pil_image, 0
    elif method == 'ocropy':
        # parameter defaults from ocropy-nlbin:
        array = pil2array(pil_image)
        bin, angle = common.binarize(array, maxskew=maxskew, threshold=threshold, nrm=nrm, zoom=zoom)
        return array2pil(bin), angle
    # equivalent to ocropy, but without deskewing:
    # elif method == 'kraken':
    #     image = kraken.binarization.nlbin(pil_image)
    #     return image, 0
    # FIXME: add 'sauvola' from OLD/ocropus-sauvola
    else:
        # Convert RGB to OpenCV
        #img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)
        img = np.asarray(pil_image.convert('L'))

        if method == 'global':
            # global thresholding
            _, th = cv2.threshold(img, threshold * 255, 255, cv2.THRESH_BINARY)
        elif method == 'otsu':
            # Otsu's thresholding
            _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'gauss-otsu':
            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise Exception('unknown binarization method %s' % method)
        return Image.fromarray(th), 0

class OcropyBinarize(Processor):
    @property
    def executable(self):
        return 'ocrd-cis-ocropy-binarize'

    def setup(self):
        method = self.parameter['method']
        if self.parameter['grayscale'] and method != 'ocropy':
            self.logger.critical(f'Requested method {method} does not support grayscale normalized output')
            raise ValueError('only method=ocropy allows grayscale=true')

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Binarize (and optionally deskew/despeckle) the pages/regions/lines of the workspace.

        Iterate over the PAGE-XML element hierarchy down to the requested
        ``level-of-operation``.

        Next, for each file, crop each segment image according to the layout
        annotation (via coordinates into the higher-level image, or from the
        alternative image), and determine the threshold for binarization and
        the deskewing angle of the segment (up to ``maxskew``). Then despeckle
        by removing connected components smaller than ``noise_maxsize``.
        Finally, apply results to the image and export it as an image file.

        Add the new image file to the workspace along with the output fileGrp,
        and using a file ID with suffix ``.IMG-BIN`` along with further
        identification of the input element.

        Reference each new image in the AlternativeImage of the element.

        Return a PAGE-XML with new AlternativeImage(s) and the arguments
        for ``workspace.save_image_file``.
        """
        level = self.parameter['level-of-operation']
        assert self.workspace
        self.logger.debug(f'Level of operation: "{level}"')

        pcgts = input_pcgts[0]
        assert pcgts
        page = pcgts.get_Page()
        assert page

        page_image, page_xywh, page_image_info = self.workspace.image_from_page(
            page, page_id, feature_filter='binarized')
        zoom = determine_zoom(self.logger, page_id, self.parameter['dpi'], page_image_info)

        result = OcrdPageResult(pcgts)
        if level == 'page':
            try:
                result.images.append(self.process_page(page, page_image, page_xywh, zoom, page_id))
            except ValueError as e:
                self.logger.error(e)
        else:
            if level == 'table':
                regions = page.get_TableRegion()
            else: # region
                regions = page.get_AllRegions(classes=['Text'], order='reading-order')
            if not regions:
                self.logger.warning(f"Page '{page_id}' contains no regions")
            for region in regions:
                region_image, region_xywh = self.workspace.image_from_segment(
                    region, page_image, page_xywh, feature_filter='binarized')
                if level == 'region':
                    try:
                        result.images.append(self.process_region(region, region_image, region_xywh, zoom, region.id))
                        continue
                    except ValueError as e:
                        self.logger.error(e)
                lines = region.get_TextLine()
                if not lines:
                    self.logger.warning(f"Page '{page_id}' region '{region.id}' contains no text lines")
                for line in lines:
                    line_image, line_xywh = self.workspace.image_from_segment(
                        line, region_image, region_xywh, feature_filter='binarized')
                    try:
                        result.images.append(self.process_line(line, line_image, line_xywh, zoom, page_id, region.id))
                    except ValueError as e:
                        self.logger.error(e)
        return result

    def process_page(self, page, page_image, page_xywh, zoom, page_id) -> OcrdPageResultImage:
        if not page_image.width or not page_image.height:
            raise ValueError(f"Skipping page '{page_id}' with zero size")
        self.logger.info(f"About to binarize page '{page_id}'")

        features = page_xywh['features']
        if 'angle' in page_xywh and page_xywh['angle']:
            # orientation has already been annotated (by previous deskewing),
            # so skip deskewing here:
            maxskew = 0
        else:
            maxskew = self.parameter['maxskew']
        bin_image, angle = binarize(
            self.logger,
            page_image,
            method=self.parameter['method'],
            maxskew=maxskew,
            threshold=self.parameter['threshold'],
            nrm=self.parameter['grayscale'],
            zoom=zoom)
        if angle:
            features += ',deskewed'
        page_xywh['angle'] = angle
        if self.parameter['noise_maxsize']:
            bin_image = remove_noise(bin_image, maxsize=self.parameter['noise_maxsize'])
            features += ',despeckled'
        # annotate angle in PAGE (to allow consumers of the AlternativeImage
        # to do consistent coordinate transforms, and non-consumers
        # to redo the rotation themselves):
        orientation = -page_xywh['angle']
        orientation = 180 - (180 - orientation) % 360  # map to [-179.999,180]
        page.set_orientation(orientation)
        if self.parameter['grayscale']:
            suffix = '.IMG-NRM'
            features += ',grayscale_normalized'
        else:
            suffix = '.IMG-BIN'
            features += ',binarized'
        # update PAGE (reference the image file):
        alt_image = AlternativeImageType(comments=features)
        page.add_AlternativeImage(alt_image)
        return OcrdPageResultImage(bin_image, suffix, alt_image)

    def process_region(self, region, region_image, region_xywh, zoom, page_id) -> OcrdPageResultImage:
        if not region_image.width or not region_image.height:
            raise ValueError(f"Skipping region '{region.id}' with zero size")
        self.logger.info(f"About to binarize page '{page_id}' region '{region.id}'")
        features = region_xywh['features']
        if 'angle' in region_xywh and region_xywh['angle']:
            # orientation has already been annotated (by previous deskewing),
            # so skip deskewing here:
            bin_image, _ = binarize(
                self.logger,
                region_image,
                method=self.parameter['method'],
                maxskew=0,
                nrm=self.parameter['grayscale'],
                zoom=zoom)
        else:
            bin_image, angle = binarize(
                self.logger,
                region_image,
                method=self.parameter['method'],
                maxskew=self.parameter['maxskew'],
                nrm=self.parameter['grayscale'],
                zoom=zoom)
            if angle:
                features += ',deskewed'
            region_xywh['angle'] = angle
        bin_image = remove_noise(bin_image, maxsize=self.parameter['noise_maxsize'])
        if self.parameter['noise_maxsize']:
            features += ',despeckled'
        # annotate angle in PAGE (to allow consumers of the AlternativeImage
        # to do consistent coordinate transforms, and non-consumers
        # to redo the rotation themselves):
        orientation = -region_xywh['angle']
        orientation = 180 - (180 - orientation) % 360 # map to [-179.999,180]
        region.set_orientation(orientation)
        suffix = f'{region.id}'
        if self.parameter['grayscale']:
            suffix += '.IMG-NRM'
            features += ',grayscale_normalized'
        else:
            suffix += '.IMG-BIN'
            features += ',binarized'
        # update PAGE (reference the image file):
        alt_image = AlternativeImageType(comments=features)
        region.add_AlternativeImage(alt_image)
        return OcrdPageResultImage(bin_image, suffix, alt_image)

    def process_line(self, line, line_image, line_xywh, zoom, page_id, region_id) -> OcrdPageResultImage:
        if not line_image.width or not line_image.height:
            raise ValueError(f"Skipping line '{line.id}' with zero size")
        self.logger.info(f"About to binarize page '{page_id}' region '{region_id}' line '{line.id}'")
        features = line_xywh['features']
        bin_image, angle = binarize(
            self.logger,
            line_image,
            method=self.parameter['method'],
            maxskew=self.parameter['maxskew'],
            nrm=self.parameter['grayscale'],
            zoom=zoom)
        if angle:
            features += ',deskewed'
        # annotate angle in PAGE (to allow consumers of the AlternativeImage
        # to do consistent coordinate transforms, and non-consumers
        # to redo the rotation themselves):
        #orientation = -angle
        #orientation = 180 - (180 - orientation) % 360 # map to [-179.999,180]
        #line.set_orientation(orientation) # does not exist on line level!
        self.logger.warning(
            f"Cannot add orientation %.2f to page '{page_id}' region '{region_id}' line '{line.id}'", -angle)
        bin_image = remove_noise(bin_image, maxsize=self.parameter['noise_maxsize'])
        if self.parameter['noise_maxsize']:
            features += ',despeckled'
        suffix = f'{region_id}_{line.id}'
        if self.parameter['grayscale']:
            suffix += '.IMG-NRM'
            features += ',grayscale_normalized'
        else:
            suffix += '.IMG-BIN'
            features += ',binarized'
        # update PAGE (reference the image file):
        alt_image = AlternativeImageType(comments=features)
        line.add_AlternativeImage(alt_image)
        return OcrdPageResultImage(bin_image, suffix, alt_image)
