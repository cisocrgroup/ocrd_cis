from __future__ import absolute_import

from ocrd_cis import get_ocrd_tool

import re
import sys
import os.path
import json
import subprocess


from ocrd_utils import getLogger, concat_padded, points_from_x0y0x1y1
from ocrd_modelfactory import page_from_file
from ocrd.model.ocrd_page import to_xml
from ocrd.model.ocrd_page import TextEquivType
from ocrd.model.ocrd_page import GlyphType
from ocrd.model.ocrd_page import WordType
from ocrd import Processor
from ocrd import MIMETYPE_PAGE


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def cmd_to_string(cmd):
    """remove unneeded whitespace from command strings"""
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
            pcgts = page_from_file(self.workspace.download_file(input_file))

            self.log.info("Processing text in page '%s'", pcgts.get_pcGtsId())
            page = pcgts.get_Page()

            index = input_file.url.rfind('/') + 1
            fgrp = input_file.url[index:-4]

            # region, line, word, or glyph level:
            regions = page.get_TextRegion()
            if not regions:
                self.log.warning("Page contains no text regions")

            self.process_regions(regions, predfiles, fgrp)

            ID = concat_padded(self.output_file_grp, n)
            self.log.info('creating file id: %s, name: %s, file_grp: %s',
                          ID, input_file.basename, self.output_file_grp)

            # Use the input file's basename for the new file
            # this way the files retain the same basenames.
            out = self.workspace.add_file(
                ID=ID,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                basename=self.output_file_grp + '-' + input_file.basename,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )
            self.log.info('created file %s', out)


    def process_regions(self, regions, predfiles, fgrp):
        for region in regions:
            self.log.info("Processing text in region '%s'", region.id)

            textlines = region.get_TextLine()
            if not textlines:
                self.log.warning(
                    "Region '%s' contains no text lines", region.id)
            else:
                self.process_lines(textlines, predfiles, fgrp, region.id)

    def process_lines(self, textlines, predfiles, fgrp, regionid):

        for line in textlines:

            for file in predfiles:
                if file == '-'.join([fgrp, regionid, line.id]):
                    self.log.info("Processing text in line '%s'", line.id)

                    filepath = self.root + '/' + file + '.json'
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

                            if char == ' ':
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
                                if i == len(positions) - 1:
                                    words.append(w)
                                    line_conf.append(word_conf)
                                    line_pos.append(word_pos)

                        wconfs = [(min(conf)+max(conf))/2 for conf in line_conf]
                        lineconf = (min(wconfs)+max(wconfs))/2

                        line.replace_TextEquiv_at(0, TextEquivType(Unicode=linepred, conf=str(lineconf)))


                        if self.maxlevel == 'word' or 'glyph':
                            box = bounding_box(line.get_Coords().points)
                            line.Word = []
                            for w_no, w in enumerate(words):

                                # Coords of word
                                wordbounding = (line_pos[w_no][0][0], line_pos[w_no][-1][-1])
                                word_bbox = [box[0]+wordbounding[0], box[1], box[2]+wordbounding[1], box[3]]

                                word_id = '%s_word%04d' % (line.id, w_no)
                                word = WordType(id=word_id, Coords=CoordsType(points_from_x0y0x1y1(word_bbox)))

                                line.add_Word(word)
                                word.add_TextEquiv(TextEquivType(Unicode=w, conf=str(wconfs[w_no])))

                                if self.maxlevel == 'glyph':
                                    for glyph_no, g in enumerate(w):
                                        glyphbounding = (line_pos[w_no][glyph_no][0], line_pos[w_no][glyph_no][-1])
                                        glyph_bbox = [box[0]+glyphbounding[0], box[1], box[2]+glyphbounding[1], box[3]]

                                        glyph_id = '%s_glyph%04d' % (word.id, glyph_no)
                                        glyph = GlyphType(id=glyph_id, Coords=CoordsType(points_from_x0y0x1y1(glyph_bbox)))

                                        word.add_Glyph(glyph)
                                        glyph.add_TextEquiv(TextEquivType(Unicode=g, conf=str(line_conf[w_no][glyph_no])))
