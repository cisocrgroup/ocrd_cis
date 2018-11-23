from __future__ import absolute_import

import sys, os.path, cv2


from ocrd.utils import getLogger, concat_padded, xywh_from_points, points_from_x0y0x1y1
from ocrd.model.ocrd_page import from_file, to_xml, TextEquivType, CoordsType, GlyphType, WordType
from ocrd.model.ocrd_page_generateds import TextStyleType, MetadataItemType, LabelsType, LabelType
from ocrd import Processor, MIMETYPE_PAGE

from ocrd_cis import get_ocrd_tool





class Stats(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-stats']
        kwargs['version'] = self.ocrd_tool['version']
        super(Stats, self).__init__(*args, **kwargs)
        self.log = getLogger('Stats')


    def process(self):
        """
        Performs the (text) recognition.
        """
        #print(self.parameter)

        maxlevel = 'line' #self.parameter['textequiv_level']


        for (n, input_file) in enumerate(self.input_files):

            pcgts = from_file(self.workspace.download_file(input_file))

            page = pcgts.get_Page()

            regions = page.get_TextRegion()

            self.process_regions(regions, maxlevel, self.input_files)




    def process_regions(self, regions, maxlevel, inputfile):
        for region in regions:

            textlines = region.get_TextLine()

            self.process_lines(textlines, maxlevel, inputfile)




    def process_lines(self, textlines, maxlevel, inputfile):

        for line in textlines:

            words = line.get_Word()
            for word in words:
                print(word.get_TextEquiv()[0].Unicode)
                print(type(word.get_TextEquiv()))
                for elem in word.get_TextEquiv():
                    print(elem.Unicode)
                    print(inputfile)
