from __future__ import absolute_import
from typing import Optional
from logging import Logger

from ocrd_utils import getLogger
from ocrd_models.ocrd_page import AlternativeImageType, OcrdPage
from ocrd import Processor, OcrdPageResult, OcrdPageResultImage

from .common import determine_zoom, remove_noise

class OcropyDenoise(Processor):
    @property
    def executable(self):
        return 'ocrd-cis-ocropy-denoise'

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Despeckle the pages / regions / lines of the workspace.

        Open and deserialise PAGE input file and its respective images,
        then iterate over the element hierarchy down to the requested
        ``level-of-operation``.

        Next, for each file, crop each segment image according to the layout
        annotation (via coordinates into the higher-level image, or from the
        alternative image). Then despeckle by removing connected components
        smaller than ``noise_maxsize``. Apply results to the image and export
        it as an image file.

        Add the new image file to the workspace along with the output fileGrp,
        and using a file ID with suffix ``.IMG-DESPECK`` along with further
        identification of the input element.

        Reference each new image in the AlternativeImage of the element.

        Produce a new output file by serialising the resulting hierarchy.
        """
        level = self.parameter['level-of-operation']
        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()

        page_image, page_xywh, page_image_info = self.workspace.image_from_page(
            page, page_id,
            feature_selector='binarized' if level == 'page' else '')
        zoom = determine_zoom(self.logger, page_id, self.parameter['dpi'], page_image_info)

        if level == 'page':
            image = self.process_segment(page, page_image, page_xywh, zoom, page_id)
            if image:
                result.images.append(image)
        else:
            regions = page.get_AllRegions(classes=['Text'], order='reading-order')
            if not regions:
                self.logger.warning(f'Page "{page_id}" contains no text regions')
            for region in regions:
                region_image, region_xywh = self.workspace.image_from_segment(
                    region, page_image, page_xywh,
                    feature_selector='binarized' if level == 'region' else '')
                if level == 'region':
                    file_id = f"{page_id}_{region.id}"
                    image = self.process_segment(region, region_image, region_xywh, zoom, file_id)
                    if image:
                        result.images.append(image)
                    continue
                lines = region.get_TextLine()
                if not lines:
                    self.logger.warning(f'Page "{page_id}" region "{region.id}" contains no text lines')
                for line in lines:
                    line_image, line_xywh = self.workspace.image_from_segment(
                        line, region_image, region_xywh, feature_selector='binarized')
                    file_id = f"{page_id}_{region.id}_{line.id}"
                    image = self.process_segment(line, line_image, line_xywh, zoom, file_id)
                    if image:
                        result.images.append(image)
        return result

    def process_segment(self, segment, segment_image, segment_xywh, zoom, file_id) -> Optional[OcrdPageResultImage]:
        if not segment_image.width or not segment_image.height:
            self.logger.warning(f"Skipping '{segment.id}' with zero size")
            return None
        self.logger.info(f"About to despeckle '{segment.id}'")
        bin_image = remove_noise(
            segment_image, maxsize=self.parameter['noise_maxsize'] / zoom * 300 / 72)  # in pt
        # update PAGE (reference the image file):
        alt_image = AlternativeImageType(comments=segment_xywh['features'] + ',despeckled')
        suffix = f"{file_id}.IMG-DESPECK"
        segment.add_AlternativeImage(alt_image)
        return OcrdPageResultImage(bin_image, suffix, alt_image)
