from __future__ import absolute_import
from ocrd.utils import getLogger, concat_padded, xywh_from_points, points_from_x0y0x1y1
from ocrd import Processor, MIMETYPE_PAGE
from ocrd_cis import get_ocrd_tool
from ocrd.model.ocrd_page_generateds import parse, parsexml_, parsexmlstring_
from difflib import SequenceMatcher



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

        cnum = 0
        tess_err = 0
        ocro1_err = 0
        ocro2_err = 0

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
                    cnum += len(gtline)

                    tessline = line.get_TextEquiv()[2].Unicode
                    ocroline1 = line.get_TextEquiv()[3].Unicode
                    ocroline2 = line.get_TextEquiv()[4].Unicode

                    s = SequenceMatcher(None, gtline, tessline)
                    tess_err += (1-s.ratio())*len(gtline)

                    s = SequenceMatcher(None, gtline, ocroline1)
                    ocro1_err += (1-s.ratio())*len(gtline)

                    s = SequenceMatcher(None, gtline, ocroline2)
                    ocro2_err += (1-s.ratio())*len(gtline)


                    # words = line.get_Word()
                    # for word in words:
                    #     for ocr in word.get_TextEquiv():
                    #         print(ocr.Unicode)


        tessac = 1-tess_err/cnum
        ocro1ac = 1-ocro1_err/cnum
        ocro2ac = 1-ocro2_err/cnum

        print('tesserocr accuracy:    ', tessac)
        print('ocropy model1 accuracy:', ocro1ac)
        print('ocropy model2 accuracy:', ocro2ac)

