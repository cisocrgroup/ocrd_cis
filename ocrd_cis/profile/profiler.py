import json
import re
from ocrd import Processor
from ocrd import MIMETYPE_PAGE
from ocrd_utils import getLogger
from ocrd_modelfactory import page_from_file
from ocrd.model.ocrd_page_generateds import TextEquivType
from ocrd.model.ocrd_page import to_xml
from ocrd_cis import get_ocrd_tool
from ocrd_cis import JavaProfiler


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
            self.workspace.add_file(
                ID="{}_{}".format(ifile.ID, self.output_file_grp),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
                file_grp=self.output_file_grp,
                pageId=ifile.pageId,
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
        """Adds candidates of the given profile to a word in the page XML
        document.  All candidates are appended to the TextEquiv of the
        given word.  New TextEquivs have
        dataType="ocrd-cis-profiler-candidate" and
        dataTypeDetails="`json of the according profiler suggestion`".

        """
        maxc = self.parameter['maxCandidates']
        i = self.parameter['index']
        _unicode = word.get_TextEquiv()[i].Unicode
        clean = re.sub(r'^\W*(.*?)\W*$', r'\1', _unicode)
        lower = clean.lower()
        if lower not in profile: # ['data']:
            return
        for cand in profile[lower]['Candidates']:
            if maxc == 0:
                break
            maxc -= 1
            eq = TextEquivType(
                dataType='ocrd-cis-profiler-candidate',
                dataTypeDetails=json.dumps(cand),
                Unicode=self.format_candidate(clean, cand['Suggestion']),
                conf=cand['Weight'],
            )
            eq.set_index(len(word.get_TextEquiv()) + 1)
            word.add_TextEquiv(eq)
            # self.log.debug("suggestion: [%s] %s (%f)",
            #                clean, eq.Unicode, cand['Weight'])

    def format_candidate(self, origin, cand):
        """
        Returns a copy of the given candidate
        with the same casing as orgin
        """
        res = ""
        for (i, c) in enumerate(cand):
            if i < len(origin) and origin[i].isupper():
                res += c.upper()
            else:
                res += c
        return res

    def read_profile(self):
        # _input = []
        # i = self.parameter['index']
        # langs = dict()
        # for (line, _, _) in self.get_all_lines():
        #     _input.append(line.get_TextEquiv()[i].Unicode)
        #     plang = line.get_primaryLanguage()
        #     if plang is not None and plang in langs:
        #         langs[line.get_primaryLanguage().lower()] += 1
        #     elif plang is not None:
        #         langs[line.get_primaryLanguage().lower()] = 1

        # lang = self.get_most_frequent_language(langs)
        # lang = "german"  # set default for now
        # p = JavaProfiler(
        #     jar=self.parameter['jar'],
        #     exe=self.parameter['executable'],
        #     backend=self.parameter['backend'],
        #     args=self.parameter['args'],
        #     lang=lang)
        # p.run("\n".join(_input))
        with open('/tmp/profile.json', encoding='utf-8') as f:
            return json.load(f)

    def get_all_lines(self):
        """Returns a list of tuples of lines, their parent and
        their according workspace file."""
        lines = []
        ifs = sorted(
            self.workspace.mets.find_files(fileGrp=self.input_file_grp),
            key=lambda ifile: ifile.ID
        )
        for ifile in ifs:
            pcgts = page_from_file(
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

    def get_most_frequent_language(self, counts):
        """ returns the most frequent language in the counts dictionary"""
        if not counts:
            return 'unknown'
        lang = sorted(counts.items(), key=lambda kv: kv[1])[0][0]
        return lang
