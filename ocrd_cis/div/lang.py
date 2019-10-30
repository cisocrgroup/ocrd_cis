from __future__ import absolute_import
from ocrd_utils import getLogger
from ocrd import Processor
from ocrd_cis import get_ocrd_tool
from ocrd.model.ocrd_page_generateds import parse
from collections import defaultdict


class Lang(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-lang']
        kwargs['version'] = self.ocrd_tool['version']
        self.input_file_grp = kwargs['input_file_grp']
        super(Lang, self).__init__(*args, **kwargs)
        self.log = getLogger('Lang')

    def process(self):
        """
        Performs the (text) recognition.
        """

        linelang = defaultdict(int)
        wordlang = defaultdict(int)

        linefont = defaultdict(int)
        wordfont = defaultdict(int)

        inputfiles = self.input_files
        for input_file in inputfiles:

            alignurl = input_file.url
            pcgts = parse(alignurl, True)
            page = pcgts.get_Page()
            regions = page.get_TextRegion()

            for region in regions:
                lines = region.get_TextLine()

                for line in lines:
                    try:
                        llang = line.primaryLanguage
                        linelang[llang] += 1
                    except TypeError:
                        pass

                    try:
                        lfont = line.fontFamily
                        linefont[lfont] += 1
                    except TypeError:
                        pass

                    words = line.get_Word()
                    for word in words:
                        try:
                            wlang = word.language
                            wordlang[wlang] += 1
                        except TypeError:
                            pass

                        try:
                            wfont = word.get_TextStyle().fontFamily
                            wordfont[wfont] += 1
                        except TypeError:
                            pass

        #predominant language
        try:
            lang = max(linelang, key=lambda k: linelang[k])
        except TypeError:
            try:
                lang = max(wordlang, key=lambda k: wordlang[k])
            except TypeError:
                lang = 'German'

        #predominant font
        try:
            font = max(linefont, key=lambda k: linefont[k])
        except TypeError:
            try:
                font = max(wordfont, key=lambda k: wordfont[k])
            except TypeError:
                font = 'Antiqua'


        print(lang)
        print(font)
