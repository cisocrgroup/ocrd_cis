from __future__ import absolute_import
from ocrd.utils import getLogger, concat_padded, xywh_from_points, points_from_x0y0x1y1
from ocrd import Processor, MIMETYPE_PAGE
from ocrd_cis import get_ocrd_tool
from ocrd.model.ocrd_page_generateds import parse, parsexml_, parsexmlstring_
from Levenshtein import distance



class Stats(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-stats']
        kwargs['version'] = self.ocrd_tool['version']
        self.input_file_grp = kwargs['input_file_grp']
        super(Stats, self).__init__(*args, **kwargs)
        self.log = getLogger('Stats')


    def process(self):
        """
        Performs the (text) recognition.
        """

        d = dict()
        if "gt" not in d:
            d["gt"]=0

        inputfiles = self.input_files
        for input_file in inputfiles:

            #index = input_file.url.rfind('/')
            #alignurl = input_file.url[:index] + '/' + self.input_file_grp + input_file.url[index:]
            alignurl = input_file.url

            pcgts = parse(alignurl, True)

            page = pcgts.get_Page()
            regions = page.get_TextRegion()


            for region in regions:
                lines = region.get_TextLine()

                for line in lines:

                    gtline = line.get_TextEquiv()[1].Unicode
                    d["gt"] += len(gtline)

                    Type = line.get_TextEquiv()[0].dataType[9:]


                    for i in range(2,len(line.get_TextEquiv())):
                        OCRType = line.get_TextEquiv()[i].dataType
                        lindex = OCRType.find('OCR-D-')
                        rindex = OCRType.find(Type)
                        model = OCRType[lindex:rindex-1]

                        
                        if model not in d:
                            d[model]=0

                        #print(line.get_TextEquiv()[2].dataType)
                        unicodeline = line.get_TextEquiv()[i].Unicode

                        d[model] += distance(gtline, unicodeline)



                        # words = line.get_Word()
                        # for word in words:
                        #     for ocr in word.get_TextEquiv():
                        #         print(ocr.Unicode)

        #tessac = 1-tess_err/cnum
        
        print(d)