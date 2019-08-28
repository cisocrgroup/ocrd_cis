from __future__ import absolute_import

from ocrd_cis.ocropy.ocrolib import lstm
from ocrd_cis.ocropy import ocrolib
from ocrd_cis import get_ocrd_tool

import sys
import os.path
import warnings
import cv2
import numpy as np
from scipy.ndimage import filters, interpolation, measurements, morphology
from scipy import stats
from PIL import Image

import Levenshtein
#import kraken.binarization

from ocrd_utils import getLogger, concat_padded, xywh_from_points, points_from_x0y0x1y1
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, TextEquivType, CoordsType, GlyphType, WordType
from ocrd_models.ocrd_page_generateds import TextStyleType, MetadataItemType, LabelsType, LabelType
from ocrd import Processor
from ocrd_utils import MIMETYPE_PAGE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LOG = getLogger('processor.OcropyRecognize')

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

# method similar to ocrolib.read_image_gray:
def pil2array(image):
    assert isinstance(image, Image.Image), "not a PIL.Image"
    array = ocrolib.pil2array(image)
    if array.dtype == np.uint8:
        array = array / 255.0
    if array.dtype == np.int8:
        array = array / 127.0
    elif array.dtype == np.uint16:
        array = array / 65536.0
    elif array.dtype == np.int16:
        array = array / 32767.0
    elif np.issubdtype(array.dtype, np.floating):
        pass
    else:
        raise Exception("unknown image type: " + array.dtype)
    if array.ndim == 3:
        array = np.mean(array, 2)
    return array

def array2pil(array):
    assert isinstance(array, np.ndarray), "not a numpy array"
    array = np.array(255.0 * array, np.uint8)
    return ocrolib.array2pil(array)

# methods from ocropy-nlbin:
def estimate_local_whitelevel(image, zoom=0.5, perc=80, range_=20):
    '''flatten it by estimating the local whitelevel
    zoom for page background estimation, smaller=faster, default: %(default)s
    percentage for filters, default: %(default)s
    range for filters, default: %(default)s
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = interpolation.zoom(image, zoom)
        m = filters.percentile_filter(m, perc, size=(range_, 2))
        m = filters.percentile_filter(m, perc, size=(2, range_))
        m = interpolation.zoom(m, 1. / zoom)
    w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    flat = np.clip(image[:w, :h] - m[:w, :h] + 1, 0, 1)
    return flat
def estimate_skew_angle(image, angles):
    estimates = []
    for a in angles:
        v = np.mean(interpolation.rotate(image, a, order=0, mode='constant'), axis=1)
        v = np.var(v)
        estimates.append((v, a))
    _, a = max(estimates)
    return a
def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8):
    ''' estimate skew angle and rotate'''
    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1) # border ignore
    flat = np.amax(flat) - flat
    flat -= np.amin(flat)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    ma = maxskew
    ms = int(2 * maxskew * skewsteps)
    # print(linspace(-ma,ma,ms+1))
    angle = estimate_skew_angle(est, np.linspace(-ma, ma, ms + 1))
    flat = interpolation.rotate(flat, angle, mode='constant', reshape=0)
    flat = np.amax(flat) - flat
    return flat, angle
def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90):
    '''# estimate low and high thresholds
    ignore this much of the border for threshold estimation, default: %(default)s
    scale for estimating a mask over the text region, default: %(default)s
    lo percentile for black estimation, default: %(default)s
    hi percentile for white estimation, default: %(default)s
    '''
    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    if escale > 0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = escale
        v = est - filters.gaussian_filter(est, e * 20.0)
        v = filters.gaussian_filter(v ** 2, e * 20.0) ** 0.5
        v = (v > 0.3 * np.amax(v))
        v = morphology.binary_dilation(v, structure=np.ones((int(e * 50), 1)))
        v = morphology.binary_dilation(v, structure=np.ones((1, int(e * 50))))
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(), lo)
    hi = stats.scoreatpercentile(est.ravel(), hi)
    return lo, hi

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

    raw_line = line.copy()
    if np.prod(line.shape) == 0:
        raise Exception('image dimensions are zero')
    if np.amax(line) == np.amin(line):
        raise Exception('image is blank')

    # dewarp:
    temp = np.amax(line)-line
    if check:
        report = check_line(temp)
        if report:
            raise Exception(report)
    if dewarp:
        temp = temp * 1.0 / np.amax(temp)
        lnorm.measure(temp)
        line = lnorm.normalize(line, cval=np.amax(line))

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

    def process(self):
        """
        Performs the (text) recognition.
        """
        # print(self.parameter)
        dewarping = self.parameter['dewarping']
        binarization = self.parameter['binarization']
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

        # LOG.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            pil_image = self.workspace.resolve_image_as_pil(
                pcgts.get_Page().imageFilename)

            LOG.info("Recognizing text in page '%s'", pcgts.get_pcGtsId())
            page = pcgts.get_Page()

            # region, line, word, or glyph level:
            regions = page.get_TextRegion()
            if not regions:
                LOG.warning("Page contains no text regions")

            args = [lnorm, network, pil_image, filepath, pad, dewarping, binarization]

            self.process_regions(regions, maxlevel, args)

            # Use the input file's basename for the new file
            # this way the files retain the same basenames.
            # ID = concat_padded(self.output_file_grp, n)
            ID = self.output_file_grp + '-' + input_file.basename.replace('.xml', '')
            LOG.info('creating file id: %s, name: %s, file_grp: %s',
                     ID, input_file.basename, self.output_file_grp)
            out = self.workspace.add_file(
                ID=ID,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                basename=input_file.basename,
                local_filename=os.path.join(self.output_file_grp, input_file.basename),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )
            LOG.info('created file %s', out)

    def process_regions(self, regions, maxlevel, args):
        edits = 0
        lengs = 0
        for region in regions:
            LOG.info("Recognizing text in region '%s'", region.id)

            textlines = region.get_TextLine()
            if not textlines:
                LOG.warning("Region '%s' contains no text lines", region.id)
            else:
                edits_, lengs_ = self.process_lines(textlines, maxlevel, args)
                edits += edits_
                lengs += lengs_
        LOG.info('CER: %.1f%%', 100.0 * edits / lengs if lengs > 0 else 0.)

    def process_lines(self, textlines, maxlevel, args):
        lnorm, network, pil_image, filepath, pad, dewarping, binarization = args

        edits = 0
        lengs = 0
        for line in textlines:
            LOG.info("Recognizing text in line '%s'", line.id)
            linegt = line.TextEquiv[0].Unicode
            LOG.debug("GT  '%s': '%s'", line.id, linegt)

            # get box from points
            if line.get_Coords().points == '':
                LOG.warn("empty bounding box at line %s", line.id)
                continue
            box = bounding_box(line.get_Coords().points)

            # crop word from page
            cropped_image = pil_image.crop(box=box)
            if cropped_image.size[1] < 32:
                LOG.warn("bounding box is too narrow at line %s", line.id)
                continue

            # binarize with Otsu's thresholding after Gaussian filtering
            bin_image = binarize(cropped_image, method=binarization)
            #bin_image.save('/tmp/ocrd-cis-ocropy-recognize_%s.bin.png' % line.id)

            # resize image to 48 pixel height
            final_img = resize_keep_ratio(bin_image)

            # process ocropy:
            try:
                linepred, clist, rlist, confidlist = process1(
                    final_img, pad, lnorm, network,
                    check=True, dewarp=dewarping)
            except Exception as e:
                LOG.error('error processing line "%s": %s', line.id, e)
                continue
            dist = Levenshtein.distance(linepred, linegt)
            edits += dist
            lengs += len(linegt)
            LOG.debug("OCR '%s': '%s'", line.id, linepred)
            LOG.debug("Distance: %d", dist)

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
        return edits, lengs
