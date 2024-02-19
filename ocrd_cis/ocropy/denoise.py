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
    to_xml, AlternativeImageType
)
from ocrd import Processor

from .. import get_ocrd_tool
from .common import (
    # binarize,
    remove_noise)

TOOL = 'ocrd-cis-ocropy-denoise'

class OcropyDenoise(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools'][TOOL]
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyDenoise, self).__init__(*args, **kwargs)

    def process(self):
        """Despeckle the pages / regions / lines of the workspace.

        Open and deserialise PAGE input files and their respective images,
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
        LOG = getLogger('processor.OcropyDenoise')
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
                
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(
                page, page_id,
                feature_selector='binarized' if level == 'page' else '')
            if self.parameter['dpi'] > 0:
                zoom = 300.0/self.parameter['dpi']
            elif page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi *= 2.54
                LOG.info('Page "%s" uses %f DPI', page_id, dpi)
                zoom = 300.0/dpi
            else:
                zoom = 1

            if level == 'page':
                self.process_segment(page, page_image, page_xywh, zoom,
                                     input_file.pageId, file_id)
            else:
                regions = page.get_AllRegions(classes=['Text'], order='reading-order')
                if not regions:
                    LOG.warning('Page "%s" contains no text regions', page_id)
                for region in regions:
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region, page_image, page_xywh,
                        feature_selector='binarized' if level == 'region' else '')
                    if level == 'region':
                        self.process_segment(region, region_image, region_xywh, zoom,
                                             input_file.pageId, file_id + '_' + region.id)
                        continue
                    lines = region.get_TextLine()
                    if not lines:
                        LOG.warning('Page "%s" region "%s" contains no text lines', page_id, region.id)
                    for line in lines:
                        line_image, line_xywh = self.workspace.image_from_segment(
                            line, region_image, region_xywh,
                            feature_selector='binarized')
                        self.process_segment(line, line_image, line_xywh, zoom,
                                             input_file.pageId,
                                             file_id + '_' + region.id + '_' + line.id)

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

    def process_segment(self, segment, segment_image, segment_xywh, zoom, page_id, file_id):
        LOG = getLogger('processor.OcropyDenoise')
        if not segment_image.width or not segment_image.height:
            LOG.warning("Skipping '%s' with zero size", file_id)
            return
        LOG.info("About to despeckle '%s'", file_id)
        bin_image = remove_noise(segment_image,
                                 maxsize=self.parameter['noise_maxsize']/zoom*300/72) # in pt
        # update METS (add the image file):
        file_path = self.workspace.save_image_file(
            bin_image, file_id + '.IMG-DESPECK', self.output_file_grp,
            page_id=page_id)
        # update PAGE (reference the image file):
        segment.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=segment_xywh['features'] + ',despeckled'))
