from __future__ import absolute_import

import sys
import os.path
import numpy as np
from PIL import Image

import Levenshtein

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_for_segment,
    polygon_from_bbox,
    points_from_polygon,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    to_xml, TextEquivType,
    CoordsType, GlyphType, WordType
)
from ocrd import Processor

from .. import get_ocrd_tool
from .ocrolib import lstm, load_object, midrange
from .common import (
    pil2array,
    check_line
)

TOOL = 'ocrd-cis-ocropy-recognize'

def resize_keep_ratio(image, baseheight=48):
    scale = baseheight / image.height
    wsize = round(image.width * scale)
    image = image.resize((wsize, baseheight), Image.LANCZOS)
    return image, scale

# from ocropus-rpred process1, but without input files and without lineest/dewarping
def recognize(image, pad, network, check=True):
    line = pil2array(image)
    binary = np.array(line <= midrange(line), np.uint8)
    raw_line = line.copy()

    # validate:
    if np.prod(line.shape) == 0:
        raise Exception('image dimensions are zero')
    if np.amax(line) == np.amin(line):
        raise Exception('image is blank')
    if check:
        report = check_line(binary)
        if report:
            raise Exception(report)

    # recognize:
    line = lstm.prepare_line(line, pad)
    pred = network.predictString(line)

    # getting confidence
    result = lstm.translate_back(network.outputs, pos=1)
    scale = len(raw_line.T)*1.0/(len(network.outputs)-2*pad)

    clist = []
    rlist = []
    confidlist = []

    for r, c in result:
        if c != 0:
            confid = network.outputs[r, c]
            c = network.l2s([c])
            r = (r-pad)*scale

            confidlist.append(confid)
            clist.append(c)
            rlist.append(r)

    return str(pred), clist, rlist, confidlist


class OcropyRecognize(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        self.pad = 16 # ocropus-rpred default
        self.network = None # set in process
        kwargs['ocrd_tool'] = self.ocrd_tool['tools'][TOOL]
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyRecognize, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()
    
    def setup(self):
        self.logger = getLogger('processor.OcropyRecognize')
        # from ocropus-rpred:
        self.network = load_object(self.get_model(), verbose=1)
        for x in self.network.walk():
            x.postLoad()
        for x in self.network.walk():
            if isinstance(x, lstm.LSTM):
                x.allocate(5000)

    def get_model(self):
        """Search for the model file.  First checks if parameter['model'] can
        be resolved with OcrdResourceManager to a valid readeable file and
        returns it.  If not, it checks if the model can be found in the
        dirname(__file__)/models/ directory."""
        canread = lambda p: os.path.isfile(p) and os.access(p, os.R_OK)
        try:
            model = self.resolve_resource(self.parameter['model'])
            if canread(model):
                return model
        except SystemExit:
            ocropydir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(ocropydir, 'models', self.parameter['model'])
            self.logger.info("Failed to resolve model with OCR-D/core mechanism, trying %s", path)
            if canread(path):
                return path
        self.logger.error("Could not find model %s. Try 'ocrd resmgr download ocrd-cis-ocropy-recognize %s",
                self.parameter['model'], self.parameter['model'])
        sys.exit(1)

    def process(self):
        """Recognize lines / words / glyphs of the workspace.

        Open and deserialise each PAGE input file and its respective image,
        then iterate over the element hierarchy down to the requested
        ``textequiv_level``. If any layout annotation below the line level
        already exists, then remove it (regardless of ``textequiv_level``).

        Set up Ocropy to recognise each text line (via coordinates into
        the higher-level image, or from the alternative image; the image
        must have been binarised/grayscale-normalised, deskewed and dewarped
        already). Rescale and pad the image, then recognize.

        Create new elements below the line level, if necessary.
        Put text results and confidence values into new TextEquiv at
        ``textequiv_level``, and make the higher levels consistent with that
        up to the line level (by concatenation joined by whitespace).

        If a TextLine contained any previous text annotation, then compare
        that with the new result by aligning characters and computing the
        Levenshtein distance. Aggregate these scores for each file and print
        the line-wise and the total character error rates (CER).

        Produce a new output file by serialising the resulting hierarchy.
        """
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        maxlevel = self.parameter['textequiv_level']

        # self.logger.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            self.logger.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()

            page_image, page_coords, _ = self.workspace.image_from_page(
                page, page_id)

            self.logger.info("Recognizing text in page '%s'", page_id)
            # region, line, word, or glyph level:
            regions = page.get_AllRegions(classes=['Text'])
            if not regions:
                self.logger.warning("Page '%s' contains no text regions", page_id)
            self.process_regions(regions, maxlevel, page_image, page_coords)

            # update METS (add the PAGE file):
            file_id = make_file_id(input_file, self.output_file_grp)
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

    def process_regions(self, regions, maxlevel, page_image, page_coords):
        edits = 0
        lengs = 0
        for region in regions:
            region_image, region_coords = self.workspace.image_from_segment(
                region, page_image, page_coords)

            self.logger.info("Recognizing text in region '%s'", region.id)
            textlines = region.get_TextLine()
            if not textlines:
                self.logger.warning("Region '%s' contains no text lines", region.id)
            else:
                edits_, lengs_ = self.process_lines(textlines, maxlevel, region_image, region_coords)
                edits += edits_
                lengs += lengs_
            # update region text by concatenation for consistency
            region_unicode = u'\n'.join(line.get_TextEquiv()[0].Unicode
                                        if line.get_TextEquiv()
                                        else u'' for line in textlines)
            region.set_TextEquiv([TextEquivType(Unicode=region_unicode)])
        if lengs > 0:
            self.logger.info('CER: %.1f%%', 100.0 * edits / lengs)

    def process_lines(self, textlines, maxlevel, region_image, region_coords):
        edits = 0
        lengs = 0
        for line in textlines:
            line_image, line_coords = self.workspace.image_from_segment(
                line, region_image, region_coords)

            self.logger.info("Recognizing text in line '%s'", line.id)
            if line.get_TextEquiv():
                linegt = line.TextEquiv[0].Unicode
            else:
                linegt = ''
            self.logger.debug("GT  '%s': '%s'", line.id, linegt)
            # remove existing annotation below line level:
            line.set_TextEquiv([])
            line.set_Word([])

            if line_image.size[1] < 16:
                self.logger.debug("ERROR: bounding box is too narrow at line %s", line.id)
                continue
            # resize image to 48 pixel height
            final_img, scale = resize_keep_ratio(line_image)

            # process ocropy:
            try:
                linepred, clist, rlist, confidlist = recognize(
                    final_img, self.pad, self.network, check=True)
            except Exception as err:
                self.logger.debug('error processing line "%s": %s', line.id, err)
                continue
            self.logger.debug("OCR '%s': '%s'", line.id, linepred)
            edits += Levenshtein.distance(linepred, linegt)
            lengs += len(linegt)

            words = [x.strip() for x in linepred.split(' ') if x.strip()]

            word_r_list = [[0]] # r-positions of every glyph in every word
            word_conf_list = [[]] # confidences of every glyph in every word
            if words != []:
                w_no = 0
                found_char = False
                for i, c in enumerate(clist):
                    if c != ' ':
                        found_char = True
                        word_conf_list[w_no].append(confidlist[i])
                        word_r_list[w_no].append(rlist[i])

                    if c == ' ' and found_char:
                        if i == 0:
                            word_r_list[0][0] = rlist[i]

                        elif i+1 <= len(clist)-1 and clist[i+1] != ' ':
                            word_conf_list.append([])
                            word_r_list.append([rlist[i]])
                            w_no += 1
            else:
                word_conf_list = [[0]]
                word_r_list = [[0, line_image.width]]

            # conf for each word
            wordsconf = [(min(x)+max(x))/2 for x in word_conf_list]
            # conf for the line
            line_conf = (min(wordsconf) + max(wordsconf))/2
            # line text
            line.add_TextEquiv(TextEquivType(
                Unicode=linepred, conf=line_conf))

            if maxlevel in ['word', 'glyph']:
                for word_no, word_str in enumerate(words):
                    word_points = points_from_polygon(
                        coordinates_for_segment(
                            np.array(polygon_from_bbox(
                                word_r_list[word_no][0] / scale,
                                0,
                                word_r_list[word_no][-1] / scale,
                                0 + line_image.height)),
                            line_image,
                            line_coords))
                    word_id = '%s_word%04d' % (line.id, word_no)
                    word = WordType(id=word_id, Coords=CoordsType(word_points))
                    line.add_Word(word)
                    word.add_TextEquiv(TextEquivType(
                        Unicode=word_str, conf=wordsconf[word_no]))

                    if maxlevel == 'glyph':
                        for glyph_no, glyph_str in enumerate(word_str):
                            glyph_points = points_from_polygon(
                                coordinates_for_segment(
                                    np.array(polygon_from_bbox(
                                        word_r_list[word_no][glyph_no] / scale,
                                        0,
                                        word_r_list[word_no][glyph_no+1] / scale,
                                        0 + line_image.height)),
                                    line_image,
                                    line_coords))
                            glyph_id = '%s_glyph%04d' % (word.id, glyph_no)
                            glyph = GlyphType(id=glyph_id, Coords=CoordsType(glyph_points))
                            word.add_Glyph(glyph)
                            glyph.add_TextEquiv(TextEquivType(
                                Unicode=glyph_str, conf=word_conf_list[word_no][glyph_no]))
        return edits, lengs
