import json
import re
from ocrd import Processor
from ocrd import MIMETYPE_PAGE
from ocrd.utils import getLogger
from ocrd.model.ocrd_page import from_file
from ocrd.model.ocrd_page_generateds import TextEquivType
from ocrd.model.ocrd_page import to_xml
from ocrd_cis import get_ocrd_tool
from ocrd_cis import JavaProcess


class Profiler(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-profile']
        kwargs['version'] = ocrd_tool['version']
        super(Profiler, self).__init__(*args, **kwargs)
        self.log = getLogger('cis.Processor.Profiler')

    def process(self):
        profile = self.read_profile()
        files = self.add_suggestions(profile)
        for (pcgts, ifile) in files:
            self.add_output_file(
                ID="{}_{}".format(ifile.ID, self.output_file_grp),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
                file_grp=self.output_file_grp,
                basename=ifile.basename,
            )
        self.workspace.save_mets()

    def add_suggestions(self, profile):
        files = []
        ids = set()
        for (word, pcgts, ifile) in self.get_all_words():
            self.add_candidates(profile, word)
            if ifile.ID not in ids:
                ids.add(ifile.ID)
                files.append((pcgts, ifile))
        return files

    def add_candidates(self, profile, word):
        i = self.parameter['textEquivIndex']
        _unicode = word.get_TextEquiv()[i].Unicode
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
        i = self.parameter['textEquivIndex']
        langs = dict()
        for (line, pcgts, ifile) in self.get_all_lines():
            _input.append(line.get_TextEquiv()[i].Unicode)
            langs[line.get_primaryLanguage().lower()] += 1

        p = JavaProcess.profiler(
            jar=self.parameter['cisOcrdJar'],
            args=[
                self.parameter['profilerExecutable'],
                self.parameter['profilerBackend'],
                self.get_most_frequent_language(langs),
            ]
        )
        return p.run("\n".join(_input))

    def get_all_lines(self):
        """Returns a list of tuples of lines, their parent and
        their according workspace file."""
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
                    lines.append((line, pcgts, ifile))
        return lines

    def get_all_words(self):
        words = []
        for (line, pcgts, ifile) in self.get_all_lines():
            for word in line.get_Word():
                words.append((word, pcgts, ifile))
        return words

    def get_most_frequent_language(counts):
        """ returns the most frequent language in the counts dictionary"""
        if counts.len() == 0:
            return 'unknown'
        lang = sorted(counts.iteritems(), key=lambda k, v: (v, k))[0][1]
        return lang
