import json
import re
from ocrd import Processor
from ocrd_cis import get_ocrd_tool
from ocrd.utils import getLogger
from ocrd.model.ocrd_page import from_file
from ocrd_cis import JavaProcess
from ocrd.model.ocrd_page_generateds import TextEquivType


class Profiler(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-profile']
        kwargs['version'] = ocrd_tool['version']
        super(Profiler, self).__init__(*args, **kwargs)
        self.log = getLogger('Processor.Profiler')

    def process(self):
        profile = self.read_profile()
        self.add_suggestions(profile)

    def add_suggestions(self, profile):
        for word in self.get_all_words():
            self.add_candidates(profile, word)

    def add_candidates(self, profile, word):
        _unicode = word.get_TextEquiv()[0].Unicode
        clean = re.sub(r'^\W*(.*?)\W*$', r'\1', _unicode)
        lower = clean.lower()
        if lower not in profile['data']:
            return
        for cand in profile['data'][lower]['Candidates']:
            eq = TextEquivType(
                dataType='profiler-candidate',
                dataTypeDetails=json.dumps(cand),
                Unicode=Profiler.format_candidate(clean, cand['Suggestion']),
                conf=cand['Weight'],
            )
            word.add_TextEquiv(eq)
            self.log.debug("suggestion: [%s] %s (%f)",
                           clean, eq.Unicode, cand['Weight'])

    def format_candidate(origin, cand):
        res = ""
        for (i, c) in enumerate(cand):
            if i < len(origin) and origin[i].isupper():
                res += c.upper()
            else:
                res += c
        return res

    def read_profile(self):
        _input = []
        for line in self.get_all_lines():
            _input.append(line.get_TextEquiv()[0].Unicode)
        p = JavaProcess(
             jar=self.parameter['cisOcrdJar'],
             args=[
                 self.parameter['profilerExecutable'],
                 self.parameter['profilerBackend'],
                 self.parameter['profilerLanguage'],
             ]
        )
        p.run_profiler("\n".join(_input))
        return json.loads(p.output)

    def get_all_lines(self):
        lines = []
        ifs = sorted(
            self.workspace.mets.find_files(fileGrp=self.input_file_grp),
            key=lambda ifile: ifile.ID
        )
        for ifile in ifs:
            pcgts = from_file(
                self.workspace.download_file(ifile)
            )
            for region in pcgts.get_Page().get_TextRegion():
                for line in region.get_TextLine():
                    lines.append(line)
        return lines

    def get_all_words(self):
        words = []
        for line in self.get_all_lines():
            for word in line.get_Word():
                words.append(word)
        return words
