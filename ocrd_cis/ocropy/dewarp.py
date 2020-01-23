from __future__ import absolute_import

import os.path
import numpy as np

from ocrd_utils import getLogger, concat_padded
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    MetadataItemType,
    LabelsType, LabelType,
    to_xml, AlternativeImageType
)
from ocrd import Processor
from ocrd_utils import MIMETYPE_PAGE

from .. import get_ocrd_tool
from .ocrolib import lineest
from .common import (
    pil2array, array2pil,
    check_line,
)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TOOL = 'ocrd-cis-ocropy-dewarp'
LOG = getLogger('processor.OcropyDewarp')
FALLBACK_FILEGRP_IMG = 'OCR-D-IMG-DEWARP'

class InvalidLine(Exception):
    """Line image does not allow dewarping and should be ignored."""

class InadequateLine(Exception):
    """Line image is not safe for dewarping and should be padded instead."""

# from ocropus-dewarp, but without resizing
def dewarp(image, lnorm, check=True, max_neighbour=0.02, zoom=1.0):
    line = pil2array(image)
    
    if np.prod(line.shape) == 0:
        raise InvalidLine('image dimensions are zero')
    if np.amax(line) == np.amin(line):
        raise InvalidLine('image is blank')
    
    temp = np.amax(line)-line # inverse, zero-closed
    if check:
        report = check_line(temp, zoom=zoom)
        if report:
            raise InadequateLine(report)
    
    temp = temp * 1.0 / np.amax(temp) # normalized
    if check:
        report = lnorm.check(temp, max_ignore=max_neighbour)
        if report:
            raise InvalidLine(report)

    lnorm.measure(temp) # find centerline
    line = lnorm.dewarp(line, cval=np.amax(line))
    
    return array2pil(line)

# pad with white above and below (as a fallback for dewarp)
def padvert(image, range_):
    line = pil2array(image)
    height = line.shape[0]
    margin = int(range_ * height / 16)
    line = np.pad(line, ((margin, margin), (0, 0)), constant_values=1.0)
    return array2pil(line)

class OcropyDewarp(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools'][TOOL]
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyDewarp, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # defaults from ocrolib.lineest:
            range_ = self.parameter['range']
            self.lnorm = lineest.CenterNormalizer(params=(range_, 1.0, 0.3))
            try:
                self.page_grp, self.image_grp = self.output_file_grp.split(',')
            except ValueError:
                self.page_grp = self.output_file_grp
                self.image_grp = FALLBACK_FILEGRP_IMG
                LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_FILEGRP_IMG)

    def process(self):
        """Dewarp the lines of the workspace.

        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the TextLine level.

        Next, get each line image according to the layout annotation (from
        the alternative image of the line, or by cropping via coordinates
        into the higher-level image), and dewarp it (without resizing).
        Export the result as an image file.

        Add the new image file to the workspace with a fileGrp USE given
        in the second position of the output fileGrp, or ``OCR-D-IMG-DEWARP``,
        and an ID based on input file and input element.

        Reference each new image in the AlternativeImage of the element.

        Produce a new output file by serialising the resulting hierarchy.
        """

        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.image_grp, n)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
                
            # add metadata about this operation and its runtime parameters:
            metadata = pcgts.get_Metadata() # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 Labels=[LabelsType(
                                     externalModel="ocrd-tool",
                                     externalId="parameters",
                                     Label=[LabelType(type_=name,
                                                      value=self.parameter[name])
                                            for name in self.parameter.keys()])]))
            
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(
                page, page_id)
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

            regions = page.get_TextRegion()
            if not regions:
                LOG.warning('Page "%s" contains no text regions', page_id)
            for region in regions:
                region_image, region_xywh = self.workspace.image_from_segment(
                    region, page_image, page_xywh)

                lines = region.get_TextLine()
                if not lines:
                    LOG.warning('Region %s contains no text lines', region.id)
                for line in lines:
                    line_image, line_xywh = self.workspace.image_from_segment(
                        line, region_image, region_xywh)

                    LOG.info("About to dewarp page '%s' region '%s' line '%s'",
                             page_id, region.id, line.id)
                    try:
                        dew_image = dewarp(line_image, self.lnorm, check=True,
                                           max_neighbour=self.parameter['max_neighbour'],
                                           zoom=zoom)
                    except InvalidLine as err:
                        LOG.error('cannot dewarp line "%s": %s', line.id, err)
                        continue
                    except InadequateLine as err:
                        LOG.warning('cannot dewarp line "%s": %s', line.id, err)
                        # as a fallback, simply pad the image vertically
                        # (just as dewarping would do on average, so at least
                        #  this line has similar margins as the others):
                        dew_image = padvert(line_image, self.parameter['range'])
                    # update METS (add the image file):
                    file_path = self.workspace.save_image_file(
                        dew_image,
                        file_id + '_' + region.id + '_' + line.id,
                        page_id=input_file.pageId,
                        file_grp=self.image_grp)
                    # update PAGE (reference the image file):
                    alternative_image = line.get_AlternativeImage()
                    line.add_AlternativeImage(AlternativeImageType(
                        filename=file_path,
                        comments=line_xywh['features'] + ',dewarped'))

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
