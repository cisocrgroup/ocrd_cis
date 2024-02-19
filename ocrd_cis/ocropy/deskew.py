from __future__ import absolute_import

import os.path

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    PageType,
    to_xml, AlternativeImageType
)
from ocrd import Processor

from .. import get_ocrd_tool
from . import common
from .common import (
    pil2array
)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TOOL = 'ocrd-cis-ocropy-deskew'

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

    def process(self):
        """Deskew the pages or regions of the workspace.

        Open and deserialise PAGE input files and their respective images,
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
        LOG = getLogger('processor.OcropyDeskew')
        level = self.parameter['level-of-operation']
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            file_id = make_file_id(input_file, self.output_file_grp)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
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
                if level == 'table':
                    regions = page.get_TableRegion()
                else: # region
                    regions = page.get_AllRegions(classes=['Text'], order='reading-order')
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
            file_path = os.path.join(self.output_file_grp, file_id + '.xml')
            pcgts.set_pcGtsId(file_id)
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

    def _process_segment(self, segment, segment_image, segment_coords, segment_id, page_id, file_id):
        LOG = getLogger('processor.OcropyDeskew')
        if not segment_image.width or not segment_image.height:
            LOG.warning("Skipping %s with zero size", segment_id)
            return
        angle0 = segment_coords['angle'] # deskewing (w.r.t. top image) already applied to segment_image
        LOG.info("About to deskew %s", segment_id)
        angle = deskew(segment_image, maxskew=self.parameter['maxskew']) # additional angle to be applied
        # segment angle: PAGE orientation is defined clockwise,
        # whereas PIL/ndimage rotation is in mathematical direction:
        orientation = -(angle + angle0)
        orientation = 180 - (180 - orientation) % 360 # map to [-179.999,180]
        segment.set_orientation(orientation) # also removes all deskewed AlternativeImages
        LOG.info("Found angle for %s: %.1f", segment_id, angle)
        # delegate reflection, rotation and re-cropping to core:
        if isinstance(segment, PageType):
            segment_image, segment_coords, _ = self.workspace.image_from_page(
                segment, page_id,
                fill='background', transparency=True)
        else:
            segment_image, segment_coords = self.workspace.image_from_segment(
                segment, segment_image, segment_coords,
                fill='background', transparency=True)
        if not angle:
            # zero rotation does not change coordinates,
            # but assures consuming processors that the
            # workflow had deskewing
            segment_coords['features'] += ',deskewed'
        # update METS (add the image file):
        file_path = self.workspace.save_image_file(
            segment_image, file_id + '.IMG-DESKEW', self.output_file_grp,
            page_id=page_id)
        # update PAGE (reference the image file):
        segment.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=segment_coords['features']))
