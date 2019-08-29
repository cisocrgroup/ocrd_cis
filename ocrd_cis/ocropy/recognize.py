from __future__ import absolute_import

import sys
import os.path
import numpy as np
from PIL import Image

import Levenshtein

from ocrd_utils import getLogger, concat_padded, xywh_from_points, points_from_xywh
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, TextEquivType, CoordsType, GlyphType, WordType
from ocrd_models.ocrd_page_generateds import TextStyleType, MetadataItemType, LabelsType, LabelType
from ocrd import Processor
from ocrd_utils import MIMETYPE_PAGE

from .. import get_ocrd_tool
from .ocrolib import lstm, load_object, midrange
from .common import (
    coordinates_for_segment,
    polygon_from_bbox,
    points_from_polygon,
    image_from_page,
    image_from_segment,
    pil2array,
    check_line
)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LOG = getLogger('processor.OcropyRecognize')

def resize_keep_ratio(image, baseheight=48):
    scale = baseheight / image.height
    wsize = round(image.width * scale)
    image = image.resize((wsize, baseheight), Image.ANTIALIAS)
    return image, scale

def binarize(pil_image, method='none'):
    if method == 'none':
        return pil_image
    # equivalent to ocropy, but without deskewing:
    # elif method == 'kraken':
    #     image = kraken.binarization.nlbin(pil_image)
    #     return image
    elif method == 'ocropy':
        # parameter defaults from ocropy-nlbin:
        args = {
            'threshold': 0.5,
            'zoom': 0.5,
            'escale': 1.0,
            'bignore': 0.1,
            'perc': 80,
            'range': 20,
            'maxskew': 2,
            'lo': 5,
            'hi': 90,
            'skewsteps': 8
        }
        # process1 from ocropy-nlbin:
        image = pil2array(pil_image)
        extreme = (np.sum(image < 0.05) + np.sum(image > 0.95)) * 1.0 / np.prod(image.shape)
        if extreme > 0.95:
            comment = "no-normalization"
            flat = image
        else:
            comment = ""
            # if not, we need to flatten it by estimating the local whitelevel
            flat = estimate_local_whitelevel(image, args['zoom'], args['perc'], args['range'])
        if args['maxskew'] > 0:
            flat, angle = estimate_skew(flat, args['bignore'], args['maxskew'], args['skewsteps'])
        else:
            angle = 0
        lo, hi = estimate_thresholds(flat, args['bignore'], args['escale'], args['lo'], args['hi'])
        # rescale the image to get the gray scale image
        flat -= lo
        flat /= (hi - lo)
        flat = np.clip(flat, 0, 1)
        bin = 1 * (flat > args['threshold'])
        #LOG.debug("binarization: lo-hi (%.2f %.2f) angle %4.1f %s", lo, hi, angle, comment)

        return array2pil(bin)
    else:
        # Convert RGB to OpenCV
        img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)

        if method == 'global':
            # global thresholding
            _, th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        elif method == 'otsu':
            # Otsu's thresholding
            _, th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif method == 'gauss-otsu':
            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            raise Exception('unknown binarization method %s', method)

        return Image.fromarray(th)


def check_line(image):
    if len(image.shape)==3: return "input image is color image %s"%(image.shape,)
    if np.mean(image)<np.median(image): return "image may be inverted"
    h,w = image.shape
    if h<20: return "image not tall enough for a text line %s"%(image.shape,)
    if h>200: return "image too tall for a text line %s"%(image.shape,)
    if w<1.2*h: return "line too short %s"%(image.shape,)
    if w>4000: return "line too long %s"%(image.shape,)
    ratio = w*1.0/h
    _,ncomps = measurements.label(image>np.mean(image))
    lo = int(0.5*ratio+0.5)
    hi = int(4*ratio)+1
    if ncomps<lo: return "too few connected components (got %d, wanted >=%d)"%(ncomps,lo)
    if ncomps>hi*ratio: return "too many connected components (got %d, wanted <=%d)"%(ncomps,hi)
    return None

def process1(image, pad, lnorm, network, check=True, dewarp=True):
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
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-ocropy-recognize']
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyRecognize, self).__init__(*args, **kwargs)

        ocropydir = os.path.dirname(os.path.abspath(__file__))
        # from ocropus-rpred:
        self.network = load_object(
            os.path.join(ocropydir, 'models', self.parameter['model']),
            verbose=1)
        for x in self.network.walk():
            x.postLoad()
        for x in self.network.walk():
            if isinstance(x, lstm.LSTM):
                x.allocate(5000)

        self.pad = 16 # ocropus-rpred default

    def process(self):
        """Recognize lines / words / glyphs of the workspace.

        Open and deserialise each PAGE input file and its respective image,
        then iterate over the element hierarchy down to the requested
        `textequiv_level`. If any layout annotation below the line level
        already exists, then remove it (regardless of `textequiv_level`).

        Set up Ocropy to recognise each text line (via coordinates into
        the higher-level image, or from the alternative image; the image
        must have been binarised/grayscale-normalised, deskewed and dewarped
        already). Rescale and pad the image, then recognize.

        Create new elements below the line level, if necessary.
        Put text results and confidence values into new TextEquiv at
        `textequiv_level`, and make the higher levels consistent with that
        up to the line level (by concatenation joined by whitespace).

        If a TextLine contained any previous text annotation, then compare
        that with the new result by aligning characters and computing the
        Levenshtein distance. Aggregate these scores for each file and print
        the line-wise and the total character error rates (CER).

        Produce a new output file by serialising the resulting hierarchy.
        """
        maxlevel = self.parameter['textequiv_level']

        # LOG.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
            page_image, page_xywh, _ = image_from_page(
                self.workspace, page, page_id)

            LOG.info("Recognizing text in page '%s'", page_id)
            # region, line, word, or glyph level:
            regions = page.get_TextRegion()
            if not regions:
                LOG.warning("Page '%s' contains no text regions", page_id)
            self.process_regions(regions, maxlevel, page_image, page_xywh)

            # update METS (add the PAGE file):
            file_id = input_file.ID.replace(self.input_file_grp,
                                            self.output_file_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.output_file_grp, n)
            file_path = os.path.join(self.output_file_grp,
                                     file_id + '.xml')
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

    def process_regions(self, regions, maxlevel, page_image, page_xywh):
        edits = 0
        lengs = 0
        for region in regions:
            region_image, region_xywh = image_from_segment(
                self.workspace, region, page_image, page_xywh)

            LOG.info("Recognizing text in region '%s'", region.id)
            textlines = region.get_TextLine()
            if not textlines:
                LOG.warning("Region '%s' contains no text lines", region.id)
            else:
                edits_, lengs_ = self.process_lines(textlines, maxlevel, region_image, region_xywh)
                edits += edits_
                lengs += lengs_
        if lengs > 0:
            LOG.info('CER: %.1f%%', 100.0 * edits / lengs)

    def process_lines(self, textlines, maxlevel, args):
        lnorm, network, pil_image, filepath, pad, dewarping, binarization = args
        edits = 0
        lengs = 0
        for line in textlines:
            line_image, line_xywh = image_from_segment(
                self.workspace, line, region_image, region_xywh)

            LOG.info("Recognizing text in line '%s'", line.id)
            if line.get_TextEquiv():
                linegt = line.TextEquiv[0].Unicode
            else:
                linegt = ''
            LOG.debug("GT  '%s': '%s'", line.id, linegt)
            # remove existing annotation below line level:
            line.set_TextEquiv([])
            line.set_Word([])

            if line_image.size[1] < 16:
                LOG.error("bounding box is too narrow at line %s", line.id)
                continue
            # resize image to 48 pixel height
            final_img = resize_keep_ratio(bin_image)

            # process ocropy:
            try:
                linepred, clist, rlist, confidlist = process1(
                    final_img, self.pad, self.network, check=True)
            except Exception as err:
                LOG.error('error processing line "%s": %s', line.id, err)
                continue
            dist = Levenshtein.distance(linepred, linegt)
            edits += dist
            lengs += len(linegt)
            LOG.debug("OCR '%s': '%s'", line.id, linepred)
            LOG.debug("Distance: %d", dist)

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
                word_r_list = [[0, line_xywh['w']]]

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
                                0 + line_xywh['h'])),
                            line_image,
                            line_xywh))
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
                                        0 + line_xywh['h'])),
                                    line_image,
                                    line_xywh))
                            glyph_id = '%s_glyph%04d' % (word.id, glyph_no)
                            glyph = GlyphType(id=glyph_id, Coords=CoordsType(glyph_points))
                            word.add_Glyph(glyph)
                            glyph.add_TextEquiv(TextEquivType(
                                Unicode=glyph_str, conf=word_conf_list[word_no][glyph_no]))
        return edits, lengs
