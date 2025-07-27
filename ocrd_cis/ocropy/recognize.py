from __future__ import absolute_import

from logging import Logger
from sys import exit
from typing import Any, Optional
from os import access, R_OK
from os.path import abspath, dirname, isfile, join
import numpy as np
from PIL import Image

from rapidfuzz.distance import Levenshtein

from ocrd_utils import coordinates_for_segment, points_from_polygon, polygon_from_bbox
from ocrd_models.ocrd_page import CoordsType, GlyphType, OcrdPage, TextEquivType, WordType
from ocrd import Processor, OcrdPageResult

from .common import check_line, pil2array
from .ocrolib import lstm, load_object, midrange


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
    result = lstm.translate_back(network.outputs, pos=1) # raw positions
    scale = len(raw_line.T) * 1.0 / (len(network.outputs) - 2 * pad)

    clist = []
    rlist = []
    confidlist = []

    for r, c in result:
        if c != 0:
            confid = network.outputs[r, c]
            c = network.l2s([c])
            r = (r - pad) * scale

            confidlist.append(confid)
            clist.append(c)
            rlist.append(r)

    return str(pred), clist, rlist, confidlist


class OcropyRecognize(Processor):
    network: Any
    pad: int

    @property
    def executable(self):
        return 'ocrd-cis-ocropy-recognize'

    def setup(self):
        self.pad = 16
        # from ocropus-rpred:
        self.network = load_object(self.get_model(), verbose=1)
        for x in self.network.walk():
            x.postLoad()
        for x in self.network.walk():
            if isinstance(x, lstm.LSTM):
                x.allocate(5000)

    def get_model(self):
        """Search for the model file.  First checks if parameter['model'] can
        be resolved with OcrdResourceManager to a valid readable file and
        returns it.  If not, it checks if the model can be found in the
        dirname(__file__)/models/ directory."""
        canread = lambda p: isfile(p) and access(p, R_OK)
        p_model = self.parameter['model']
        try:
            model = self.resolve_resource(p_model)
            if canread(model):
                return model
        except SystemExit:
            ocropydir = dirname(abspath(__file__))
            path = join(ocropydir, 'models', p_model)
            self.logger.info(f"Failed to resolve model with OCR-D/core mechanism, trying {path}")
            if canread(path):
                return path
        self.logger.error(
            f"Could not find model {p_model}. Try 'ocrd resmgr download ocrd-cis-ocropy-recognize {p_model}")
        exit(1)

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: str = None) -> OcrdPageResult:
        """Recognize lines / words / glyphs of a page.

        Open and deserialize the PAGE input file and its respective image,
        then iterate over the element hierarchy down to the requested
        ``textequiv_level``. If any layout annotation below the line level
        already exists, then remove it (regardless of ``textequiv_level``).

        Set up Ocropy to recognize each text line (via coordinates into
        the higher-level image, or from the alternative image; the image
        must have been binarized/grayscale-normalised, deskewed and dewarped
        already). Rescale and pad the image, then recognize.

        Create new elements below the line level, if necessary.
        Put text results and confidence values into new TextEquiv at
        ``textequiv_level``, and make the higher levels consistent with that
        up to the line level (by concatenation joined by whitespace).

        If a TextLine contained any previous text annotation, then compare
        that with the new result by aligning characters and computing the
        Levenshtein distance. Aggregate these scores for each file and print
        the line-wise and the total character error rates (CER).

        Return the resulting OcrdPage.
        """
        max_level = self.parameter['textequiv_level']
        assert self.workspace
        self.logger.debug(f'Max level: "{max_level}"')

        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        assert page

        page_image, page_xywh, _ = self.workspace.image_from_page(page, page_id)
        self.logger.info(f"Recognizing text in page '{page_id}'")
        # region, line, word, or glyph level:
        regions = page.get_AllRegions(classes=['Text'])
        if not regions:
            self.logger.warning(f"Page '{page_id}' contains no text regions")
        self.process_regions(regions, max_level, page_image, page_xywh)
        return OcrdPageResult(pcgts)

    def process_regions(self, regions, maxlevel, page_image, page_xywh):
        edits = 0
        lengs = 0
        for region in regions:
            region_image, region_xywh = self.workspace.image_from_segment(region, page_image, page_xywh)
            self.logger.info(f"Recognizing text in region '{region.id}'")
            textlines = region.get_TextLine()
            if not textlines:
                self.logger.warning(f"Region '{region.id}' contains no text lines")
            else:
                edits_, lengs_ = self.process_lines(textlines, maxlevel, region_image, region_xywh)
                edits += edits_
                lengs += lengs_
            # update region text by concatenation for consistency
            region_unicode = u'\n'.join(
                line.get_TextEquiv()[0].Unicode if line.get_TextEquiv() else u'' for line in textlines)
            region.set_TextEquiv([TextEquivType(Unicode=region_unicode)])
        if lengs > 0:
            self.logger.info('CER: %.1f%%', 100.0 * edits / lengs)

    def process_lines(self, textlines, maxlevel, region_image, region_xywh):
        edits = 0
        lengs = 0
        for line in textlines:
            line_image, line_coords = self.workspace.image_from_segment(line, region_image, region_xywh)
            self.logger.info(f"Recognizing text in line '{line.id}'")
            if line.get_TextEquiv():
                linegt = line.TextEquiv[0].Unicode
            else:
                linegt = ''
            self.logger.debug(f"GT  '{line.id}': '{linegt}'")
            # remove existing annotation below line level:
            line.set_TextEquiv([])
            line.set_Word([])

            if line_image.size[1] < 16:
                self.logger.debug(f"Error: bounding box is too narrow at line {line.id}")
                continue
            # resize image to 48 pixel height
            final_img, scale = resize_keep_ratio(line_image)

            # process ocropy:
            try:
                linepred, clist, rlist, confidlist = recognize(final_img, self.pad, self.network, check=True)
            except Exception as err:
                self.logger.debug(f'Error processing line "{line.id}": {str(err) or err.__class__.__name__}')
                continue
            self.logger.debug(f"OCR '{line.id}': '{linepred}'")
            edits += Levenshtein.distance(linepred, linegt)
            lengs += len(linegt)

            words = [x.strip() for x in linepred.split(' ') if x.strip()]

            word_r_list = [[0]]  # r-positions of every glyph in every word
            word_conf_list = [[]]  # confidences of every glyph in every word
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
                        elif i + 1 <= len(clist) - 1 and clist[i + 1] != ' ':
                            word_conf_list.append([])
                            word_r_list.append([rlist[i]])
                            w_no += 1
            else:
                word_conf_list = [[0]]
                word_r_list = [[0, line_image.width]]

            # conf for each word
            wordsconf = [(min(x) + max(x)) / 2 for x in word_conf_list]
            # conf for the line
            line_conf = (min(wordsconf) + max(wordsconf)) / 2
            # line text
            line.add_TextEquiv(TextEquivType(Unicode=linepred, conf=line_conf))

            if maxlevel in ['word', 'glyph']:
                for word_no, word_str in enumerate(words):
                    word_points = points_from_polygon(
                        coordinates_for_segment(
                            np.array(polygon_from_bbox(
                                word_r_list[word_no][0] / scale,0,
                                word_r_list[word_no][-1] / scale, 0 + line_image.height)),
                            line_image,
                            line_coords))
                    word_id = '%s_word%04d' % (line.id, word_no)
                    word = WordType(id=word_id, Coords=CoordsType(word_points))
                    line.add_Word(word)
                    word.add_TextEquiv(TextEquivType(Unicode=word_str, conf=wordsconf[word_no]))

                    if maxlevel == 'glyph':
                        for glyph_no, glyph_str in enumerate(word_str):
                            glyph_points = points_from_polygon(
                                coordinates_for_segment(
                                    np.array(polygon_from_bbox(
                                        word_r_list[word_no][glyph_no] / scale, 0,
                                        word_r_list[word_no][glyph_no + 1] / scale, 0 + line_image.height)),
                                    line_image,
                                    line_coords))
                            glyph_id = '%s_glyph%04d' % (word.id, glyph_no)
                            glyph = GlyphType(id=glyph_id, Coords=CoordsType(glyph_points))
                            word.add_Glyph(glyph)
                            glyph.add_TextEquiv(
                                TextEquivType(Unicode=glyph_str, conf=word_conf_list[word_no][glyph_no]))
        return edits, lengs
