from __future__ import absolute_import

import os.path
import numpy as np

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
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

class InvalidLine(Exception):
    """Line image does not allow dewarping and should be ignored."""

class InadequateLine(Exception):
    """Line image is not safe for dewarping and should be padded instead."""

# from ocropus-dewarp, but without resizing
def dewarp(image, lnorm, check=True, max_neighbour=0.02, zoom=1.0):
    if not image.width or not image.height:
        raise InvalidLine('image size is zero')
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
            # processing context
            self.setup()
    
    def setup(self):
        # defaults from ocrolib.lineest:
        self.lnorm = lineest.CenterNormalizer(
            params=(self.parameter['range'],
                    self.parameter['smoothness'],
                    # let's not expose this for now
                    # (otherwise we must explain mutual
                    #  dependency between smoothness
                    #  and extra params)
                    0.3))
        self.logger = getLogger('processor.OcropyDewarp')

    def process(self):
        """Dewarp the lines of the workspace.

        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the TextLine level.

        Next, get each line image according to the layout annotation (from
        the alternative image of the line, or by cropping via coordinates
        into the higher-level image), and dewarp it (without resizing).
        Export the result as an image file.

        Add the new image file to the workspace along with the output fileGrp,
        and using a file ID with suffix ``.IMG-DEWARP`` along with further
        identification of the input element.

        Reference each new image in the AlternativeImage of the element.

        Produce a new output file by serialising the resulting hierarchy.
        """
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        for (n, input_file) in enumerate(self.input_files):
            self.logger.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            file_id = make_file_id(input_file, self.output_file_grp)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
                
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(
                page, page_id)
            if self.parameter['dpi'] > 0:
                zoom = 300.0/self.parameter['dpi']
            elif page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi *= 2.54
                self.logger.info('Page "%s" uses %f DPI', page_id, dpi)
                zoom = 300.0/dpi
            else:
                zoom = 1

            regions = page.get_AllRegions(classes=['Text'], order='reading-order')
            if not regions:
                self.logger.warning('Page "%s" contains no text regions', page_id)
            for region in regions:
                region_image, region_xywh = self.workspace.image_from_segment(
                    region, page_image, page_xywh)

                lines = region.get_TextLine()
                if not lines:
                    self.logger.warning('Region %s contains no text lines', region.id)
                for line in lines:
                    line_image, line_xywh = self.workspace.image_from_segment(
                        line, region_image, region_xywh)

                    self.logger.info("About to dewarp page '%s' region '%s' line '%s'",
                                     page_id, region.id, line.id)
                    try:
                        dew_image = dewarp(line_image, self.lnorm, check=True,
                                           max_neighbour=self.parameter['max_neighbour'],
                                           zoom=zoom)
                    except InvalidLine as err:
                        self.logger.error('cannot dewarp line "%s": %s', line.id, err)
                        continue
                    except InadequateLine as err:
                        self.logger.warning('cannot dewarp line "%s": %s', line.id, err)
                        # as a fallback, simply pad the image vertically
                        # (just as dewarping would do on average, so at least
                        #  this line has similar margins as the others):
                        dew_image = padvert(line_image, self.parameter['range'])
                    # update METS (add the image file):
                    file_path = self.workspace.save_image_file(
                        dew_image,
                        file_id + '_' + region.id + '_' + line.id + '.IMG-DEWARP',
                        self.output_file_grp,
                        page_id=input_file.pageId)
                    # update PAGE (reference the image file):
                    alternative_image = line.get_AlternativeImage()
                    line.add_AlternativeImage(AlternativeImageType(
                        filename=file_path,
                        comments=line_xywh['features'] + ',dewarped'))

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
            self.logger.info('created file ID: %s, file_grp: %s, path: %s',
                             file_id, self.output_file_grp, out.local_filename)
