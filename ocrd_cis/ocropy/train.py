from __future__ import absolute_import

import sys
import os
import tempfile

from ocrd_modelfactory import page_from_file
from ocrd import Processor
from ocrd_utils import getLogger
from ocrd_cis import get_ocrd_tool

from .ocropus_rtrain import *
from .binarize import binarize


def deletefiles(filelist):
    for file in filelist:
        if os.path.exists(file):
            os.remove(file)
        if os.path.exists(file[:-3]+'gt.txt'):
            os.remove(file[:-3]+'gt.txt')

def resize_keep_ratio(image, baseheight=48):
    hpercent = (baseheight / float(image.size[1]))
    wsize = int((float(image.size[0] * float(hpercent))))
    image = image.resize((wsize, baseheight), Image.LANCZOS)
    return image


class OcropyTrain(Processor):

    def __init__(self, *args, **kwargs):
        self.oldcwd = os.getcwd()
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-ocropy-train']
        kwargs['version'] = ocrd_tool['version']
        super(OcropyTrain, self).__init__(*args, **kwargs)
        if hasattr(self, 'input_file_grp'):
            # processing context
            self.setup()

    def setup(self):
        self.log = getLogger('processor.OcropyTrain')
        #print(self.parameter)
        if 'model' in self.parameter:
            model = self.parameter['model']
            try:
                modelpath = self.resolve_resource(model)
            except SystemExit:
                ocropydir = os.path.dirname(os.path.abspath(__file__))
                modelpath = os.path.join(ocropydir, 'models', model)
                self.log.info("Failed to resolve model '%s' path, trying '%s'", model, modelpath)
            if not os.path.isfile(modelpath):
                self.log.error("Could not find model '%s'. Try 'ocrd resmgr download ocrd-cis-ocropy-recognize %s'",
                               model, model)
                sys.exit(1)
            outputpath = os.path.join(self.oldcwd, 'output', model)
            if 'outputpath' in self.parameter:
                outputpath = os.path.join(self.parameter, model)
        else:
            modelpath = None
            outputpath = os.path.join(self.oldcwd, 'output', 'lstm')
            if 'outputpath' in self.parameter:
                outputpath = os.path.join(self.parameter, 'lstm')
        os.makedirs(os.path.dirname(outputpath))
        self.modelpath = modelpath
        self.outputpath = outputpath

    def process(self):
        """
        Trains a new model on the text lines from the input fileGrp,
        extracted as temporary image-text file pairs.
        """
        filelist = []
        filepath = tempfile.mkdtemp(prefix='ocrd-cis-ocropy-train-')
        #self.log.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            #self.log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
            page_image, page_coords, _ = self.workspace.image_from_page(page, page_id)

            self.log.info("Extracting from page '%s'", page_id)
            for region in page.get_AllRegions(classes=['Text']):
                textlines = region.get_TextLine()
                self.log.info("Extracting %i lines from region '%s'", len(textlines), region.id)
                for line in textlines:
                    if self.parameter['textequiv_level'] == 'line':
                        path = os.path.join(filepath, page_id + region.id + line.id)
                        imgpath = self.extract_segment(path, line, page_image, page_coords)
                        if imgpath:
                            filelist.append(imgpath)
                        continue
                    for word in line.get_Word():
                        if self.parameter['textequiv_level'] == 'word':
                            path = os.path.join(filepath, page_id + region.id + line.id + word.id)
                            imgpath = self.extract_segment(path, word, page_image, page_coords)
                            if imgpath:
                                filelist.append(imgpath)
                            continue
                        for glyph in word.get_Glyph():
                            path = os.path.join(filepath, page_id + region.id + line.id + glyph.id)
                            imgpath = self.extract_segment(path, glyph, page_image, page_coords)
                            if imgpath:
                                filelist.append(imgpath)

        self.log.info("Training %s from %s on %i file pairs",
                      self.outputpath,
                      self.modelpath or 'scratch',
                      len(filelist))
        rtrain(filelist, self.modelpath, self.outputpath, self.parameter['ntrain'])
        deletefiles(filelist)

    def extract_segment(self, path, segment, page_image, page_coords):
        #ground truth
        gt = segment.TextEquiv
        if not gt:
            return None
        gt = gt[0].Unicode
        if not gt or not gt.strip():
            return None
        gt = gt.strip()
        gtpath = path + '.gt.txt'
        with open(gtpath, "w", encoding='utf-8') as f:
            f.write(gt)

        self.log.debug("Extracting %s '%s'", segment.__class__.__name__, segment.id)
        image, coords = self.workspace.image_from_segment(segment, page_image, page_coords)

        if 'binarized' not in coords['features'].split(','):
            # binarize with nlbin
            image, _ = binarize(image, maxskew=0)

        # resize image to 48 pixel height
        image = resize_keep_ratio(image)

        #save temp image
        imgpath = path + '.png'
        image.save(imgpath)

        return imgpath
