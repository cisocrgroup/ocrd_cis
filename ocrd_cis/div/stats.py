from __future__ import absolute_import
import json
from ocrd_utils import getLogger
from ocrd import Processor
from ocrd_cis import get_ocrd_tool
from ocrd.model.ocrd_page_generateds import parse
from Levenshtein import distance


class Stats(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-stats']
        kwargs['version'] = self.ocrd_tool['version']
        self.input_file_grp = kwargs['input_file_grp']
        super(Stats, self).__init__(*args, **kwargs)
        self.log = getLogger('Stats')

    def get_gt_index(self, regions):
        for region in regions:
            for line in region.get_TextLine():
                i = 0
                for te in line.get_TextEquiv():
                    ddt = te.get_dataTypeDetails()
                    if ddt.startswith('OCR-D-GT'):
                        return i
                    i += 1
        return -1

    def process(self):
        """
        Performs the (text) recognition.
        """

        d = dict()
        # if "gt" not in d:
        #     d["gt"] = 0

        inputfiles = self.input_files
        for input_file in inputfiles:

            # index = input_file.url.rfind('/')
            # alignurl = input_file.url[:index] + '/' + self.input_file_grp + input_file.url[index:]
            alignurl = input_file.url

            pcgts = parse(alignurl, True)

            page = pcgts.get_Page()
            regions = page.get_TextRegion()

            # find index of GT
            gti = self.get_gt_index(regions)
            if gti == -1:
                # sigh. just give up for this file
                continue
            if gti not in d:
                d['gt'] = 0

            for region in regions:
                lines = region.get_TextLine()

                for line in lines:
                    if len(line.get_TextEquiv()) <= gti:
                        continue
                    gtline = line.get_TextEquiv()[gti].Unicode
                    d['gt'] += len(gtline)

                    #Type = line.get_TextEquiv()[0].dataType[9:]

                    # for i in range(2, len(line.get_TextEquiv())):
                    for i in range(0, gti):
                        # print("%d: %s".format(i, line.get_
                        # OCRType = line.get_TextEquiv()[i].dataType
                        # lindex = OCRType.find('OCR-D-')
                        # rindex = OCRType.find(Type)
                        # model = OCRType[lindex:rindex-1]

                        if i not in d:
                            d[i] = 0
                        # print(line.get_TextEquiv()[2].dataType)
                        unicodeline = line.get_TextEquiv()[i].Unicode

                        d[i] += distance(gtline, unicodeline)

                        # words = line.get_Word()
                        # for word in words:
                        #     for ocr in word.get_TextEquiv():
                        #         print(ocr.Unicode)

        print(json.dumps(d))
