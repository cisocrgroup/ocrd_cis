from __future__ import absolute_import

from ocrd_cis.ocropy.ocrolib import lstm
from ocrd_cis.ocropy import ocrolib
from ocrd_cis import get_ocrd_tool

import re
import sys
import os.path
import cv2
import json
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


def bounding_box(coord_points):
    point_list = [[int(p) for p in pair.split(',')]
                  for pair in coord_points.split(' ')]
    x_coordinates, y_coordinates = zip(*point_list)
    return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))


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
        self.maxlevel = self.parameter['textequiv_level']
        linesdir = self.parameter['linesdir']



        if self.maxlevel not in ['line', 'word', 'glyph']:
            raise Exception(
                "currently only implemented at the line/glyph level")

        root, _, files = os.walk(linesdir).__next__()
        self.root = root
        predfiles = []
        for file in files:
            if '.pred' in file:
                predfiles.append(file[:-9])



########################################################################################



        # self.log.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            # self.log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = from_file(self.workspace.download_file(input_file))

            self.log.info("Recognizing text in page '%s'", pcgts.get_pcGtsId())
            page = pcgts.get_Page()

            index = input_file.url.rfind('/') + 1
            fgrp = input_file.url[index:-4]
            fgrplen = len(fgrp)


            regionfiles = []
            for file in predfiles:
                if len(file) >= fgrplen and fgrp in file[:fgrplen]:
                    regionfiles.append(file)

            # region, line, word, or glyph level:
            regions = page.get_TextRegion()
            if not regions:
                self.log.warning("Page contains no text regions")

            self.process_regions(regions, regionfiles, fgrplen)

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

    def process_regions(self, regions, regionfiles, fgrplen):
        for region in regions:
            self.log.info("Recognizing text in region '%s'", region.id)
            regionlen = len(region.id)

            textlines = region.get_TextLine()
            if not textlines:
                self.log.warning(
                    "Region '%s' contains no text lines", region.id)
            else:
                linefiles = []
                frlen = fgrplen + regionlen
                for file in regionfiles:
                    if len(file) >= frlen and region.id in file[fgrplen:regionlen]:
                        linefiles.append(f)

                self.process_lines(textlines, linefiles, frlen)

    def process_lines(self, textlines, linefiles, frlen):

        for line in textlines:
            self.log.info("Recognizing text in line '%s'", line.id)
            lidlen = line.id


            for file in linefiles:
                if len(file) >= lidlen + frlen and line.id in file:

                    filepath = self.root + '/' + file + '/' + '.json'
                    with open(filepath) as f:
                        data = json.load(f)
                        
                        linepred = data['predictions'][0]['sentence']
                        line_conf = []
                        line_pos = []

                        w = ''
                        word_conf = []
                        words = []
                        word_pos = []

                        positions = data['predictions'][0]['positions']
                        for i, d in enumerate(positions):
                            char = d['chars'][0]['char']
                            char_conf = d['chars'][0]['probability']
                            char_pos = (d['globalStart'], d['globalEnd'])


                            if char == ' ' or i == len(positions)-1:
                                words.append(w)
                                w = ''
                                line_conf.append(word_conf)
                                word_conf = []
                                line_pos.append(word_pos)
                                word_pos = []
                            else:
                                w += char
                                word_conf.append(char_conf)
                                word_pos.append(char_pos)



                        wconfs = [(min(conf)+max(conf))/2 for conf in line_conf]
                        lineconf = [(min(wconfs)+max(wconfs))/2]

                        line.replace_TextEquiv_at(0, TextEquivType(Unicode=linepred, conf=lineconf))

                        if self.maxlevel == 'word' or 'glyph':
                            box = bounding_box(line.get_Coords().points)
                            line.Word = []
                            for w_no, w in enumerate(words):

                                # Coords of word
                                wordbounding = (line_pos[w_no][0][0], line_pos[w_no][-1][-1])
                                word_bbox = [box[0]+wordbounding[0], box[1], box[2]+wordbounding[1], box[3]]

                                word_id = '%s_word%04d' % (line.id, w_no)
                                word = WordType(id=word_id, Coords=CoordsType(
                                    points_from_x0y0x1y1(word_bbox)))

                                line.add_Word(word)
                                word.add_TextEquiv(TextEquivType(
                                    Unicode=w, conf=wconfs[w_no]))
