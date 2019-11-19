from __future__ import absolute_import

import os.path

from ocrd_utils import (
    getLogger, concat_padded,
    rotate_image,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, AlternativeImageType
from ocrd import Processor

from .. import get_ocrd_tool
from . import common
from .common import (
    pil2array
)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TOOL = 'ocrd-cis-ocropy-deskew'
LOG = getLogger('processor.OcropyDeskew')
FALLBACK_FILEGRP_IMG = 'OCR-D-IMG-DESKEW'

def deskew(pil_image, maxskew=2):
    array = pil2array(pil_image)
    _, angle = common.binarize(array, maxskew=maxskew)
    return angle

class OcropyDeskew(Processor):

    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools'][TOOL]
        kwargs['version'] = ocrd_tool['version']
        super(OcropyDeskew, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    def process(self):
        """Deskew the regions of the workspace.

        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the TextRegion level.

        Next, for each file, crop each region image according to the layout
        annotation (via coordinates into the higher-level image, or from the
        alternative image), and determine the threshold for binarization and
        the deskewing angle of the region (up to ``maxskew``). Annotate the
        angle in the region.

        Add a new image file to the workspace with the fileGrp USE given
        in the second position of the output fileGrp, or ``OCR-D-IMG-DESKEW``,
        and an ID based on input file and input element.

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
            page_image, page_coords, _ = self.workspace.image_from_page(
                page, page_id,
                # image must not have been rotated already,
                # (we will overwrite @orientation anyway,)
                # abort if no such image can be produced:
                feature_filter='deskewed' if level == 'page' else '')
            if level == 'page':
                self._process_segment(page, page_image, page_coords,
                                      "page '%s'" % page_id, input_file.pageId,
                                      file_id)
            else:
                regions = page.get_TextRegion()
                if not regions:
                    LOG.warning('Page "%s" contains no text regions', page_id)
                for region in regions:
                    # process region:
                    region_image, region_coords = self.workspace.image_from_segment(
                        region, page_image, page_coords,
                        # image must not have been rotated already,
                        # (we will overwrite @orientation anyway,)
                        # abort if no such image can be produced:
                        feature_filter='deskewed')
                    self._process_segment(region, region_image, region_coords,
                                          "region '%s'" % region.id, input_file.pageId,
                                          file_id + '_' + region.id)

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

    def _process_segment(self, segment, segment_image, segment_coords, segment_id, page_id, file_id):
        features = segment_coords['features'] # features already applied to segment_image
        angle0 = segment_coords['angle'] # deskewing (w.r.t. top image) already applied to segment_image
        LOG.info("About to deskew %s", segment_id)
        angle = deskew(segment_image, maxskew=self.parameter['maxskew']) # additional angle to be applied
        # segment angle: PAGE orientation is defined clockwise,
        # whereas PIL/ndimage rotation is in mathematical direction:
        orientation = -(angle + angle0)
        orientation = 180 - (180 - orientation) % 360 # map to [-179.999,180]
        segment.set_orientation(orientation)
        LOG.info("Found angle for %s: %.1f", segment_id, angle)
        if angle:
            LOG.debug("Rotating segment '%s' by %.2f°",
                      segment_id, angle)
            segment_image = rotate_image(segment_image, angle,
                                         fill='background', transparency=True)
            features += ',deskewed'
        # update METS (add the image file):
        file_path = self.workspace.save_image_file(
            segment_image,
            file_id,
            page_id=page_id,
            file_grp=self.image_grp)
        # update PAGE (reference the image file):
        segment.add_AlternativeImage(AlternativeImageType(
            filename=file_path, comments=features))
