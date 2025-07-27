from __future__ import absolute_import
from typing import Optional
from logging import Logger

from ocrd_utils import getLogger
from ocrd_models.ocrd_page import AlternativeImageType, OcrdPage, PageType
from ocrd import Processor, OcrdPageResult, OcrdPageResultImage

from . import common
from .common import pil2array

def deskew(pil_image, maxskew=2):
    array = pil2array(pil_image)
    _, angle = common.binarize(array, maxskew=maxskew)
    return angle

class OcropyDeskew(Processor):
    @property
    def executable(self):
        return 'ocrd-cis-ocropy-deskew'

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Deskew the pages or regions of the workspace.

        Open and deserialise PAGE input file and its respective images,
        then iterate over the element hierarchy down to the TextRegion level.

        Next, for each file, crop each region image according to the layout
        annotation (via coordinates into the higher-level image, or from the
        alternative image), and determine the threshold for binarization and
        the deskewing angle of the region (up to ``maxskew``). Annotate the
        angle in the region.

        Add the new image file to the workspace along with the output fileGrp,
        and using a file ID with suffix ``.IMG-DESKEW`` along with further
        identification of the input element.

        Produce a new output file by serialising the resulting hierarchy.
        """
        level = self.parameter['level-of-operation']
        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()

        page_image, page_coords, _ = self.workspace.image_from_page(
            page, page_id,
            # image must not have been rotated already,
            # (we will overwrite @orientation anyway,)
            # abort if no such image can be produced:
            feature_filter='deskewed' if level == 'page' else '')
        if level == 'page':
            image = self._process_segment(page, page_image, page_coords, "page '%s'" % page_id, page_id)
            if image:
                result.images.append(image)
            return result
        if level == 'table':
            regions = page.get_TableRegion()
        else:  # region
            regions = page.get_AllRegions(classes=['Text'], order='reading-order')
        if not regions:
            self.logger.warning('Page "%s" contains no text regions', page_id)
        for region in regions:
            # process region:
            region_image, region_coords = self.workspace.image_from_segment(
                region, page_image, page_coords,
                # image must not have been rotated already,
                # (we will overwrite @orientation anyway,)
                # abort if no such image can be produced:
                feature_filter='deskewed')
            image = self._process_segment(region, region_image, region_coords, f"region '{region.id}'", page_id)
            if image:
                result.images.append(image)
        return result

    def _process_segment(
            self, segment, segment_image, segment_coords, segment_id, page_id
    ) -> Optional[OcrdPageResultImage]:
        if not segment_image.width or not segment_image.height:
            self.logger.warning("Skipping %s with zero size", segment_id)
            return None
        angle0 = segment_coords['angle']  # deskewing (w.r.t. top image) already applied to segment_image
        self.logger.info(f"About to deskew {segment_id}")
        angle = deskew(segment_image, maxskew=self.parameter['maxskew'])  # additional angle to be applied
        # segment angle: PAGE orientation is defined clockwise,
        # whereas PIL/ndimage rotation is in mathematical direction:
        orientation = -(angle + angle0)
        orientation = 180 - (180 - orientation) % 360  # map to [-179.999,180]
        segment.set_orientation(orientation)  # also removes all deskewed AlternativeImages
        self.logger.info(f"Found angle for {segment_id}: %.1f", angle)
        # delegate reflection, rotation and re-cropping to core:
        if isinstance(segment, PageType):
            segment_image, segment_coords, _ = self.workspace.image_from_page(
                segment, page_id, fill='background', transparency=True)
            suffix = '.IMG-DESKEW'
        else:
            segment_image, segment_coords = self.workspace.image_from_segment(
                segment, segment_image, segment_coords, fill='background', transparency=True)
            suffix = segment.id + '.IMG-DESKEW'
        if not angle:
            # zero rotation does not change coordinates,
            # but assures consuming processors that the
            # workflow had deskewing
            segment_coords['features'] += ',deskewed'
        # update PAGE (reference the image file):
        alternative = AlternativeImageType(comments=segment_coords['features'])
        segment.add_AlternativeImage(alternative)
        return OcrdPageResultImage(segment_image, suffix, alternative)
