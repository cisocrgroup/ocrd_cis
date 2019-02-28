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
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, TextEquivType, CoordsType, GlyphType, WordType
from ocrd_models.ocrd_page_generateds import TextStyleType, MetadataItemType, LabelsType, LabelType
from ocrd_utils import MIMETYPE_PAGE
from ocrd import Processor


sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def process1(fname, pad, lnorm, network):
    base, _ = ocrolib.allsplitext(fname)
    line = ocrolib.read_image_gray(fname)

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


def OcropyRec(fromdir, todir, model):


    """
    Performs the (text) recognition.
    """
    # print(self.parameter)

    filepath = os.path.dirname(os.path.abspath(__file__))

    ocropydir = os.path.dirname(os.path.abspath(__file__))
    network = ocrolib.load_object(
        os.path.join(ocropydir, 'models', model),
        verbose=1)
    for x in network.walk():
        x.postLoad()
    for x in network.walk():
        if isinstance(x, lstm.LSTM):
            x.allocate(5000)
    lnorm = getattr(network, "lnorm", None)

    pad = 16  # default: 16



    _, dirs, _ = os.walk(fromdir).__next__()

    pngs = []
    for dir in dirs:
        root, _, files = os.walk(fromdir + dir).__next__()

        for file in files:
            if '.png' in file[-4:]:
                pngs.append(root+'/'+file)

    # self.log.info("Using model %s in %s for recognition", model)


    for (n, png) in enumerate(pngs):

        image = Image.open(png)
        w, _ = image.size

        #(h, w = image.shape)
        if w > 5000:
            print("final image too long: %d", w)
            continue

        # process ocropy
        linepred, clist, rlist, confidlist = process1(png, pad, lnorm, network)
        print(linepred)

        with open(png[:-4] + '--ocropy-' + model + '.txt', 'w+') as outf:
            outf.write(linepred)
