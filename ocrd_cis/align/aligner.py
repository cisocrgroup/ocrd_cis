from __future__ import absolute_import
import json
import os
import Levenshtein
from ocrd import Processor
from ocrd import MIMETYPE_PAGE
from ocrd.utils import getLogger
from ocrd.utils import concat_padded
from ocrd.model.ocrd_page import from_file
from ocrd.model.ocrd_page import to_xml
from ocrd.model.ocrd_page_generateds import TextEquivType
from ocrd.model.ocrd_page_generateds import parse
from ocrd_cis import JavaAligner
from ocrd_cis import get_ocrd_tool


class Aligner(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-align']
        kwargs['version'] = ocrd_tool['version']
        super(Aligner, self).__init__(*args, **kwargs)
        self.log = getLogger('cis.Processor.Aligner')

    def process(self):
        ifgs = self.input_file_grp.split(",")  # input file groups
        if len(ifgs) < 2:
            raise Exception("need at least two input file groups to align")
        ifts = self.zip_input_files(ifgs)  # input file tuples
        for _id, ift in enumerate(ifts):
            alignments = json.loads(self.run_java_aligner(ift))
            pcgts = self.align(alignments, ift)
            ID = concat_padded(self.output_file_grp, _id+1)
            basename = os.path.basename(ift[0].input_file.url)
            out = self.workspace.add_file(
                ID=ID,
                file_grp=self.output_file_grp,
                basename=self.output_file_grp + '-' + basename,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )
            self.log.info('created file %s', out)

    def align(self, alignments, ift):
        """align the alignment objects with the according input file tuples"""
        for t in ift:
            self.log.info("tuple %s", os.path.basename(t.input_file.url))
        pcgtst = self.open_input_file_tuples(ift)
        i = 0
        for mi, mr in enumerate(pcgtst[0].get_Page().get_TextRegion()):
            for mj, _ in enumerate(mr.get_TextLine()):
                lines = []
                for ii, t in enumerate(ift):
                    tr = pcgtst[ii].get_Page().get_TextRegion()
                    region = tr[mi].get_TextLine()[mj]
                    lines.append(Alignment(t, region, alignments[i]))
                self.align_lines(lines)
                i += 1
        return pcgtst[0]

    def align_lines(self, lines):
        """align the given line alignment with the lines"""
        for i, line in enumerate(lines):
            self.log.info('line alignment: %s [%s - %s]',
                          line.region.get_TextEquiv()[0].Unicode,
                          line.region.get_id(),
                          line.input_file.input_file_group)
            if i != 0:
                lines[0].region.add_TextEquiv(line.region.get_TextEquiv()[0])
                lines[0].region.get_TextEquiv()[i].set_dataType(
                    "ocrd-cis-line-alignment")
            else:
                lines[0].region.get_TextEquiv()[i].set_dataType(
                    "ocrd-cis-line-alignment-master-ocr")
            lines[0].region.get_TextEquiv()[i].set_comments(
                line.input_file.input_file_group + "/" + line.region.get_id())
        self.align_words(lines)

    def align_words(self, lines):
        self.log.info(json.dumps(lines[0].alignment))
        mregion = lines[0].region.get_Word()
        oregion = [lines[i].region.get_Word() for i in range(1, len(lines))]

        for word in lines[0].alignment['words']:
            self.log.debug("aligning word %s", word['master'])
            master, rest = self.find_word([word['master']], mregion, "master")
            mregion = rest
            if master is None or len(master) != 1:
                raise Exception("cannot find {}".format(word['master']))
            others = list()
            for i, other in enumerate(word['alignments']):
                match, rest = self.find_word(other, oregion[i])
                if match is None:
                    raise Exception("cannot find {}".format(other))
                others.append(match)
                oregion[i] = rest
            words = list()
            words.append(
                Alignment(lines[0].input_file, master, lines[0].alignment))
            for i, other in enumerate(others):
                words.append(Alignment(
                    lines[i+1].input_file,
                    other,
                    lines[i+1].alignment))
            self.align_word_regions(words)

    def align_word_regions(self, words):
        def te0(x):
            return x.get_TextEquiv()[0]
        for i, word in enumerate(words):
            _str = " ".join([te0(x).Unicode for x in word.region])
            _id = ",".join([x.get_id() for x in word.region])
            # if conf is none it is most likely ground truth data
            conf = min([te0(x).get_conf() or 1 for x in word.region])
            ifg = word.input_file.input_file_group
            self.log.info("word alignment: %s [%s - %s]", _str, _id, ifg)
            ddt = word.input_file.input_file_group + "/" + _id
            if i != 0:
                te = TextEquivType(
                    Unicode=_str,
                    conf=conf,
                    dataType="ocrd-cis-word-alignment",
                    dataTypeDetails=ddt)
                words[0].region[0].add_TextEquiv(te)
            else:
                words[0].region[0].get_TextEquiv()[i].set_dataType(
                    'ocrd-cis-word-alignment-master-ocr')
                words[0].region[0].get_TextEquiv()[i].set_dataTypeDetails(ddt)
            words[0].region[0].get_TextEquiv()[i].set_index(i+1)

    def find_word(self, tokens, regions, t="other"):
        self.log.debug("tokens = %s [%s]", tokens, t)
        for i, _ in enumerate(regions):
            n = self.match_tokens(tokens, regions, i)
            if n == 0:
                continue
            return tuple([regions[i:n], regions[i:]])
        # not found try again with levenshtein
        self.log.warn(
            "could not find tokens = %s [%s]; trying again",
            tokens, t)
        for i, _ in enumerate(regions):
            n = self.match_tokens_lev(tokens, regions, i)
            if n == 0:
                continue
            return tuple([regions[i:n], regions[i:]])
        # nothing could be found
        return tuple([None, regions])

    def match_tokens(self, tokens, regions, i):
        def f(a, b): return a in b
        return self.match_tokens_lambda(tokens, regions, i, f)

    def match_tokens_lev(self, tokens, regions, i):
        def f(a, b):
            k = int(len(a)/3)
            d = Levenshtein.distance(a, b)
            return d <= 1 or d <= k
        return self.match_tokens_lambda(tokens, regions, i, f)

    def match_tokens_lambda(self, tokens, regions, i, f):
        """
        Returns one after the last index of the match starting from i.
        Returns 0 if nothing could be matched.
        """
        for j, token in enumerate(tokens):
            if j + i > len(regions):
                return 0
            self.log.debug('checking %s with %s', token,
                           regions[i+j].get_TextEquiv()[0].Unicode)
            if f(token, regions[i+j].get_TextEquiv()[0].Unicode):
                continue
            if j == 0:
                return 0
            # skip this and try next one
            # if we already have found a
            # match ath the first token position
            i += 1
        return i + len(tokens)

    def open_input_file_tuples(self, ift):
        """
        opens all xml files of the given input file tuple
        and returns them as tuples
        """
        res = list()
        for ifile in ift:
            pcgts = ifile.open()
            res.append(pcgts)
        return tuple(res)

    def zip_input_files(self, ifgs):
        """Zip files of the given input file groups"""
        files = list()
        for ifg in ifgs:
            self.log.info("input file group: %s", ifg)
            ifiles = sorted(
                self.workspace.mets.find_files(fileGrp=ifg),
                key=lambda ifile: ifile.url)
            for i in ifiles:
                self.log.info("sorted file: %s %s",
                              os.path.basename(i.url), i.ID)
            ifiles = [FileAlignment(x, ifg) for x in ifiles]
            files.append(ifiles)
        return zip(*files)

    def read_lines_from_input_file(self, ifile):
        self.log.info("reading input file: %s", ifile)
        lines = list()
        pcgts = ifile.open()
        for region in pcgts.get_Page().get_TextRegion():
            for line in region.get_TextLine():
                lines.append(line.get_TextEquiv()[0].Unicode)
        return lines

    def run_java_aligner(self, ifs):
        lines = list()
        for ifile in ifs:
            lines.append(self.read_lines_from_input_file(ifile))
        lines = zip(*lines)
        _input = [x.strip() for t in lines for x in t]
        for i in _input:
            self.log.debug("input line: %s", i)
        n = len(ifs)
        p = JavaAligner(self.parameter['cisOcrdJar'], n)
        return p.run("\n".join(_input))


class FileAlignment:
    def __init__(self, ifile, ifg):
        self.input_file = ifile
        self.input_file_group = ifg
        self.log = getLogger('cis.FileAlignment')

    def open(self):
        self.log.info("opening: %s", os.path.basename(self.input_file.url))
        return parse(self.input_file.url, True)


class Alignment:
    def __init__(self, ifile, region, alignment):
        self.input_file = ifile
        self.region = region
        self.alignment = alignment
