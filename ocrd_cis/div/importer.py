from __future__ import absolute_import

from ocrd_cis.ocropy.ocrolib import lstm
from ocrd_cis.ocropy import ocrolib
from ocrd_cis import get_ocrd_tool

import re
import sys
import os.path
import cv2
import numpy as np
from PIL import Image
import subprocess


from ocrd.utils import getLogger, concat_padded, xywh_from_points, points_from_x0y0x1y1
from ocrd.model.ocrd_page import from_file, to_xml, TextEquivType, CoordsType, GlyphType, WordType
from ocrd.model.ocrd_page_generateds import TextStyleType, MetadataItemType, LabelsType, LabelType
from ocrd import Processor, MIMETYPE_PAGE


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def cmd_to_string(cmd):
    """remove unneded whitepsace from command strings"""
    return re.sub("""\\s+""", " ", cmd).strip()

def subprocess_cmd(command, want=0):
    print("running command: {}".format(cmd_to_string(command)))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, _ = process.communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))
    returncode = process.wait()
    if returncode != want:
        raise Exception("invalid returncode for {cmd}: {c}"
                        .format(cmd=cmd_to_string(command), c=returncode))




class Importer(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-importer']
        kwargs['version'] = self.ocrd_tool['version']
        super(Importer, self).__init__(*args, **kwargs)
        self.log = getLogger('Importer')

    def process(self):
        """
        Performs the (text) recognition.
        """

        # print(self.parameter)
        maxlevel = self.parameter['textequiv_level']
        linesdir = self.parameter['linesdir']



        if maxlevel not in ['line', 'word', 'glyph']:
            raise Exception(
                "currently only implemented at the line/glyph level")

        root, _, files = os.walk(linesdir).__next__()
        preds = {}
        for file in files:
            if '.pred' in file:
                with open(root + '/' + file) as f:
                    line = f.readline()
                    preds[file[:-9]] = line


########################################################################################



        # self.log.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            # self.log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = from_file(self.workspace.download_file(input_file))
            pil_image = self.workspace.resolve_image_as_pil(
                pcgts.get_Page().imageFilename)

            self.log.info("Recognizing text in page '%s'", pcgts.get_pcGtsId())
            page = pcgts.get_Page()

            index = input_file.url.rfind('/') + 1
            fgrp = input_file.url[index:-4]
            regionpreds = {}
            for pred in preds:
                if fgrp in pred:
                    regionpreds[pred] = preds[pred]

            # region, line, word, or glyph level:
            regions = page.get_TextRegion()
            if not regions:
                self.log.warning("Page contains no text regions")

            self.process_regions(regions, maxlevel, regionpreds)

            ID = concat_padded(self.output_file_grp, n)
            self.log.info('creating file id: %s, name: %s, file_grp: %s',
                          ID, input_file.basename, self.output_file_grp)
            # Use the input file's basename for the new file
            # this way the files retain the same basenames.
            out = self.workspace.add_file(
                ID=ID,
                file_grp=self.output_file_grp,
                basename=self.output_file_grp + '-' + input_file.basename,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )
            self.log.info('created file %s', out)

    def process_regions(self, regions, maxlevel, regionpreds):
        for region in regions:
            self.log.info("Recognizing text in region '%s'", region.id)

            textlines = region.get_TextLine()
            if not textlines:
                self.log.warning(
                    "Region '%s' contains no text lines", region.id)
            else:
                linepreds = {}
                for pred in regionpreds:
                    if region.id in pred:
                        linepreds[pred] = regionpreds[pred]

                self.process_lines(textlines, maxlevel, linepreds)

    def process_lines(self, textlines, maxlevel, linepreds):

        for line in textlines:
            self.log.info("Recognizing text in line '%s'", line.id)

            linepred = ''
            for pred in linepreds:
                if line.id in pred:
                    linepred = linepreds[pred]

            line.replace_TextEquiv_at(0, TextEquivType(
                Unicode=linepred, conf=1))
            # Todo: confidence from calamari?

            # words = [x.strip() for x in linepred.split(' ') if x.strip()]
            # if maxlevel == 'word' or 'glyph':
            #     line.Word = []
            #     for word in words:
            #
            #         line.add_Word(word)
            #         word.add_TextEquiv(TextEquivType(Unicode=word, conf=1))

