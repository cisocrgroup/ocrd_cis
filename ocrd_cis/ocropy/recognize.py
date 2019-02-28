from __future__ import absolute_import

from ocrd_cis.ocropy.ocrolib import lstm
from ocrd_cis.ocropy import ocrolib
from ocrd_cis import get_ocrd_tool

import sys
import os.path
import cv2
import numpy as np
from PIL import Image

from ocrd_utils import getLogger, concat_padded, xywh_from_points, points_from_x0y0x1y1
from ocrd_models.ocrd_page import page_from_file
from ocrd.model.ocrd_page import to_xml, TextEquivType, CoordsType, GlyphType, WordType
from ocrd.model.ocrd_page_generateds import TextStyleType, MetadataItemType, LabelsType, LabelType
from ocrd import Processor, MIMETYPE_PAGE


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def bounding_box(coord_points):
    point_list = [[int(p) for p in pair.split(',')]
                  for pair in coord_points.split(' ')]
    x_coordinates, y_coordinates = zip(*point_list)
    return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))


def resize_keep_ratio(image, baseheight=48):
    hpercent = (baseheight / float(image.size[1]))
    wsize = int((float(image.size[0] * float(hpercent))))
    image = image.resize((wsize, baseheight), Image.ANTIALIAS)
    return image


def binarize(pil_image):
    # Convert RGB to OpenCV
    img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)

    # global thresholding
    # ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    bin_img = Image.fromarray(th3)
    return bin_img


def deletefile(file):
    if os.path.exists(file):
        os.remove(file)


def process1(fname, pad, lnorm, network):
    base, _ = ocrolib.allsplitext(fname)
    line = ocrolib.read_image_gray(fname)
    deletefile(fname)

    raw_line = line.copy()
    if np.prod(line.shape) == 0:
        return None
    if np.amax(line) == np.amin(line):
        return None

    temp = np.amax(line)-line
    temp = temp*1.0/np.amax(temp)
    lnorm.measure(temp)
    line = lnorm.normalize(line, cval=np.amax(line))

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
        self.log = getLogger('OcropyRecognize')

    def process(self):
        """
        Performs the (text) recognition.
        """
        # print(self.parameter)
        maxlevel = self.parameter['textequiv_level']
        if maxlevel not in ['line', 'word', 'glyph']:
            raise Exception(
                "currently only implemented at the line/glyph level")

        filepath = os.path.dirname(os.path.abspath(__file__))

        ocropydir = os.path.dirname(os.path.abspath(__file__))
        network = ocrolib.load_object(
            os.path.join(ocropydir, 'models', self.parameter['model']),
            verbose=1)
        for x in network.walk():
            x.postLoad()
        for x in network.walk():
            if isinstance(x, lstm.LSTM):
                x.allocate(5000)
        lnorm = getattr(network, "lnorm", None)

        pad = 16  # default: 16

        # self.log.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            # self.log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            pil_image = self.workspace.resolve_image_as_pil(
                pcgts.get_Page().imageFilename)

            self.log.info("Recognizing text in page '%s'", pcgts.get_pcGtsId())
            page = pcgts.get_Page()

            # region, line, word, or glyph level:
            regions = page.get_TextRegion()
            if not regions:
                self.log.warning("Page contains no text regions")

            args = [lnorm, network, pil_image, filepath, pad]

            self.process_regions(regions, maxlevel, args)

            # Use the input file's basename for the new file
            # this way the files retain the same basenames.
            # ID = concat_padded(self.output_file_grp, n)
            ID = self.output_file_grp + '-' + input_file.basename.replace('.xml', '')
            self.log.info('creating file id: %s, name: %s, file_grp: %s',
                          ID, input_file.basename, self.output_file_grp)
            out = self.workspace.add_file(
                ID=ID,
                file_grp=self.output_file_grp,
                basename=self.output_file_grp + '-' + input_file.basename,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )
            self.log.info('created file %s', out)

    def process_regions(self, regions, maxlevel, args):
        for region in regions:
            self.log.info("Recognizing text in region '%s'", region.id)

            textlines = region.get_TextLine()
            if not textlines:
                self.log.warning(
                    "Region '%s' contains no text lines", region.id)
            else:
                self.process_lines(textlines, maxlevel, args)

    def process_lines(self, textlines, maxlevel, args):
        lnorm, network, pil_image, filepath, pad = args

        for line in textlines:
            self.log.info("Recognizing text in line '%s'", line.id)

            # get box from points
            if line.get_Coords().points == '':
                self.log.warn("empty bounding box")
                continue
            box = bounding_box(line.get_Coords().points)

            # crop word from page
            croped_image = pil_image.crop(box=box)

            # binarize with Otsu's thresholding after Gaussian filtering
            # bin_image = binarize(croped_image)
            bin_image = croped_image

            # resize image to 48 pixel height
            final_img = resize_keep_ratio(bin_image)
            w, _ = final_img.size
            if w > 5000:
                self.log.warn("final image too long: %d", w)
                continue
            # print("w = {}, h = {}".format(w, h))
            # final_img.save('/tmp/foo.png')
            # save temp image
            imgpath = os.path.join(filepath, 'temp/temp.png')
            final_img.save(imgpath)

            # process ocropy
            linepred, clist, rlist, confidlist = process1(
                imgpath, pad, lnorm, network)

            words = [x.strip() for x in linepred.split(' ') if x.strip()]

            # lists in list for every word with r-position and confidence of each glyph
            word_r_list = [[0]]
            word_conf_list = [[]]

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
                word_r_list = [[0, box[2]-box[0]]]

            # conf for each word
            wordsconf = [(min(x)+max(x))/2 for x in word_conf_list]

            # conf for the line
            line_conf = (min(wordsconf) + max(wordsconf))/2

            line.replace_TextEquiv_at(0, TextEquivType(
                Unicode=linepred, conf=line_conf))

            if maxlevel == 'word' or 'glyph':
                line.Word = []
                for w_no, w in enumerate(words):

                    # Coords of word
                    wr = (word_r_list[w_no][0], word_r_list[w_no][-1])
                    word_bbox = [box[0]+wr[0], box[1], box[2]+wr[1], box[3]]

                    word_id = '%s_word%04d' % (line.id, w_no)
                    word = WordType(id=word_id, Coords=CoordsType(
                        points_from_x0y0x1y1(word_bbox)))

                    line.add_Word(word)
                    word.add_TextEquiv(TextEquivType(
                        Unicode=w, conf=wordsconf[w_no]))

                    if maxlevel == 'glyph':
                        for glyph_no, g in enumerate(w):
                            gr = (word_r_list[w_no][glyph_no],
                                  word_r_list[w_no][glyph_no+1])
                            glyph_bbox = [box[0]+gr[0],
                                          box[1], box[2]+gr[1], box[3]]

                            glyph_id = '%s_glyph%04d' % (word.id, glyph_no)
                            glyph = GlyphType(id=glyph_id, Coords=CoordsType(
                                points_from_x0y0x1y1(glyph_bbox)))

                            word.add_Glyph(glyph)
                            glyph.add_TextEquiv(TextEquivType(
                                Unicode=g, conf=word_conf_list[w_no][glyph_no]))
