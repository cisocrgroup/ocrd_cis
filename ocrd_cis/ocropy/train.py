from __future__ import absolute_import

from typing import Optional
from logging import Logger
from sys import exit
from os import makedirs, remove
from os.path import abspath, dirname, exists, join, isfile

from ocrd_models import OcrdPage
from ocrd import Processor, Workspace, OcrdPageResult

from .ocropus_rtrain import *
from .binarize import binarize


def deletefiles(filelist):
    for file in filelist:
        if exists(file):
            remove(file)
        if exists(file[:-3] + 'gt.txt'):
            remove(file[:-3] + 'gt.txt')

def resize_keep_ratio(image, baseheight=48):
    hpercent = (baseheight / float(image.size[1]))
    wsize = int((float(image.size[0] * float(hpercent))))
    image = image.resize((wsize, baseheight), Image.LANCZOS)
    return image


class OcropyTrain(Processor):
    modelpath: str
    outputpath: str

    @property
    def executable(self):
        return 'ocrd-cis-ocropy-train'

    def setup(self):
        if 'model' in self.parameter:
            model = self.parameter['model']
            try:
                self.modelpath = self.resolve_resource(model)
            except SystemExit:
                ocropydir = dirname(abspath(__file__))
                self.modelpath = join(ocropydir, 'models', model)
                self.logger.error(f"Failed to resolve model '{model}' path, trying '{self.modelpath}'")
            if not isfile(self.modelpath):
                self.logger.critical(f"Could not find model '{model}'.\n"
                                     f"Try 'ocrd resmgr download ocrd-cis-ocropy-recognize {model}'")
                exit(1)
            self.outputpath = join(self.parameter.get('outputpath', 'output'), model)
        else:
            self.modelpath = None
            self.outputpath = join(self.parameter.get('outputpath', 'output'), 'lstm')
        makedirs(dirname(self.outputpath))
        self.filelist = None

    def process_workspace(self, workspace: Workspace) -> None:
        """
        Trains a new model on the text lines from the input fileGrp,
        extracted as image-text file pairs into the output fileGrp.
        (If the output fileGrp already exists and these files should
        be re-used, pass the `--overwrite` option when processing.)

        The model is written into `outputpath` (or just `output`) under
        the same name as `model` (i.e. the start model, or just `lstm`).
        """
        self.filelist = []
        super().process_workspace(workspace)
        self.logger.info(f"Training {self.outputpath} from {self.modelpath or 'scratch'} "
                         f"on {len(self.filelist)} file pairs")
        rtrain(self.filelist, self.modelpath, self.outputpath, self.parameter['ntrain'])
        # deletefiles(self.filelist)

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """
        Extracts pairs of plaintext and cropped image files for each text line
        in the PAGE file (to be used during training).
        """
        pcgts = input_pcgts[0]
        #self.logger.info("Using model %s in %s for recognition", model)
        page = pcgts.get_Page()
        page_image, page_coords, _ = self.workspace.image_from_page(page, page_id)

        self.logger.debug(f"Extracting from page '{page_id}'")
        for region in page.get_AllRegions(classes=['Text']):
            textlines = region.get_TextLine()
            self.logger.debug(f"Extracting {len(textlines)} lines from region '{region.id}'")
            for line in textlines:
                if self.parameter['textequiv_level'] == 'line':
                    path = join(self.output_file_grp, f"{page_id}_{region.id}_{line.id}")
                    self.filelist.append(self.extract_segment(path, line, page_image, page_coords))
                    continue
                for word in line.get_Word():
                    if self.parameter['textequiv_level'] == 'word':
                        path = join(self.output_file_grp, f"{page_id}_{region.id}_{line.id}_{word.id}")
                        self.filelist.append(self.extract_segment(path, word, page_image, page_coords))
                        continue
                    for glyph in word.get_Glyph():
                        path = join(self.output_file_grp, f"{page_id}_{region.id}_{line.id}_{word.id}_{glyph.id}")
                        self.filelist.append(self.extract_segment(path, glyph, page_image, page_coords))
        # FIXME: PAGE-XML not really needed, find a way around this (raising special exception?)
        return OcrdPageResult(pcgts)

    def extract_segment(self, path, segment, page_image, page_coords):
        gtpath = path + '.gt.txt'
        imgpath = path + '.png'
        if exists(gtpath) and exists(imgpath):
            self.logger.debug(f"Reusing {segment.__class__.__name__} '{segment.id}' file pair")
            return imgpath

        gt = segment.TextEquiv
        if not gt:
            return None
        gt = gt[0].Unicode
        if not gt or not gt.strip():
            return None
        gt = gt.strip()
        with open(gtpath, "w", encoding='utf-8') as f:
            f.write(gt)

        self.logger.debug(f"Extracting {segment.__class__.__name__} '{segment.id}' file pair")
        image, coords = self.workspace.image_from_segment(segment, page_image, page_coords)

        if 'binarized' not in coords['features'].split(','):
            # binarize with nlbin
            image, _ = binarize(self.logger, image, maxskew=0)

        # resize image to 48 pixel height
        image = resize_keep_ratio(image)

        image.save(imgpath)

        return imgpath
