from __future__ import absolute_import

import os.path

from ocrd_utils import (
    getLogger,
    concat_padded,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, AlternativeImageType
from ocrd import Processor

from .. import get_ocrd_tool
from .common import (
    # binarize,
    remove_noise)

LOG = getLogger('processor.OcropyDenoise')
FALLBACK_FILEGRP_IMG = 'OCR-D-IMG-DESPECK'

class OcropyDenoise(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-ocropy-denoise']
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyDenoise, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

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
        
        Add the new image file to the workspace with the fileGrp USE given
        in the second position of the output fileGrp, or ``OCR-D-IMG-DESPECK``,
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
            
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(
                page, page_id,
                feature_selector='binarized' if level == 'page' else '')
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
                LOG.info('Page "%s" uses %d DPI', page_id, dpi)
                zoom = 300.0/dpi
            else:
                zoom = 1
            
            if level == 'page':
                self.process_segment(page, page_image, page_xywh, zoom,
                                     input_file.pageId, file_id)
            else:
                regions = page.get_TextRegion() + (
                    page.get_TableRegion() if level == 'region' else [])
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
    
    def process_segment(self, segment, segment_image, segment_xywh, zoom, page_id, file_id):
        LOG.info("About to despeckle '%s'", file_id)
        bin_image = remove_noise(segment_image,
                                 maxsize=self.parameter['noise_maxsize']/zoom*300/72) # in pt
        # update METS (add the image file):
        file_path = self.workspace.save_image_file(
            bin_image,
            file_id,
            page_id=page_id,
            file_grp=self.image_grp)
        # update PAGE (reference the image file):
        segment.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=segment_xywh['features'] + ',despeckled'))
        
