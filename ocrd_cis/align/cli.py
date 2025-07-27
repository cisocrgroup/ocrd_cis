from __future__ import absolute_import
from __future__ import annotations

import click
import json
import os
from typing import Optional, List, Dict, Type

from rapidfuzz.distance import Levenshtein

from ocrd import Processor, OcrdPage, OcrdPageResult
from ocrd.decorators import ocrd_cli_options
from ocrd.decorators import ocrd_cli_wrap_processor
from ocrd_utils import getLevelName
from ocrd_models.ocrd_page import TextRegionType, TextEquivType
from ocrd_cis import JavaAligner


@click.command()
@ocrd_cli_options
def ocrd_cis_align(*args, **kwargs):
    return ocrd_cli_wrap_processor(CISAligner, *args, **kwargs)

class CISAligner(Processor):
    @property
    def executable(self):
        return 'ocrd-cis-align'

    def process_page_pcgts(self, *input_pcgts : Optional[OcrdPage], page_id : Optional[str] = None) -> OcrdPageResult:
        assert len(input_pcgts) >= 2
        alignments = json.loads(self.run_java_aligner(input_pcgts))
        pcgts = self.align(alignments, input_pcgts)
        return OcrdPageResult(pcgts)

    def align(self, alignments: List[Dict], pcgts: List[OcrdPage]) -> OcrdPage:
        """align the alignment objects with the according input file tuples"""
        i = 0
        file_groups = self.input_file_grp.split(',')
        for mi, mr in enumerate(pcgts[0].get_Page().get_AllRegions(classes=['Text'])):
            for mj, _ in enumerate(mr.get_TextLine()):
                lines = []
                for ii, page in enumerate(pcgts):
                    if i >= len(alignments):
                        break
                    tr = page.get_Page().get_AllRegions(classes=['Text'])
                    region = tr[mi].get_TextLine()[mj]
                    lines.append(Alignment(file_groups[ii], page, region, alignments[i]))
                self.align_lines(lines)
                i += 1
        return pcgts[0]

    def align_lines(self, lines: List[Alignment]) -> None:
        """align the given line alignment with the lines"""
        if not lines:
            return
        if len(lines[0].region.TextEquiv) > 1:
            del lines[0].region.TextEquiv[1:]
        for i, line in enumerate(lines):
            if lines[0].region.get_TextEquiv() is None:
                lines[0].region.TextEquiv = []
            self.logger.debug(
                'line alignment: %s [%s - %s]',
                get_textequiv_unicode(line.region),
                line.region.get_id(),
                line.file_grp
            )
            ddt = line.file_grp + "/" + line.region.get_id()
            if i > 0:
                te = TextEquivType(
                    Unicode=get_textequiv_unicode(line.region),
                    conf=get_textequiv_conf(line.region),
                    dataType="other",
                    dataTypeDetails=f"ocrd-cis-line-alignment:{ddt}")
                lines[0].region.add_TextEquiv(te)
            else:
                self.logger.debug("len: %i, i: %i", len(lines[0].region.TextEquiv), i)
                lines[0].region.TextEquiv[i].set_dataType("other")
                lines[0].region.TextEquiv[i].set_dataTypeDetails(
                    "ocrd-cis-line-alignment-master-ocr:" + ddt)
            lines[0].region.TextEquiv[i].set_index(i+1)
        self.align_words(lines)

    def align_words(self, lines: List[Alignment]) -> None:
        # self.logger.info(json.dumps(lines[0].alignment))
        mregion = lines[0].region.get_Word()
        oregion = [lines[i].region.get_Word() for i in range(1, len(lines))]
        for word in lines[0].alignment['wordAlignments']:
            self.logger.debug("aligning word %s", word['master'])
            master, rest = self.find_word([word['master']], mregion, "master")
            mregion = rest
            if master is None or len(master) != 1:
                self.logger.warn("cannot find {}; giving up".format(word['master']))
                # raise Exception("cannot find {}; giving up".format(word['master']))
                return
            others = list()
            for i, other in enumerate(word['alignments']):
                match, rest = self.find_word(other, oregion[i])
                if match is None:
                    self.logger.warn(f"cannot find {other}; giving up")
                    return
                others.append(match)
                oregion[i] = rest
            words = list()
            words.append(
                Alignment(lines[0].file_grp, lines[0].pcgts, master, lines[0].alignment))
            for i, other in enumerate(others):
                words.append(Alignment(
                    lines[i+1].file_grp,
                    lines[i+1].pcgts,
                    other,
                    lines[i+1].alignment))
            self.align_word_regions(words)

    def align_word_regions(self, words: List[Alignment]) -> None:
        def te0(x):
            return x.TextEquiv[0]
        for i, word in enumerate(words):
            if not word.region:
                ifg = word.file_grp
                self.logger.debug("(empty) word alignment: [%s]", ifg)
                te = TextEquivType(
                    dataType="other",
                    dataTypeDetails="ocrd-cis-empty-word-alignment:" + ifg)
                words[0].region[0].add_TextEquiv(te)
                words[0].region[0].get_TextEquiv()[i].set_index(i+1)
                continue
            _str = " ".join([te0(x).Unicode for x in word.region])
            _id = ",".join([x.get_id() for x in word.region])
            ifg = word.file_grp
            ddt = word.file_grp + "/" + _id
            # if conf is none it is most likely ground truth data
            conf = min([float(te0(x).get_conf() or "1.0") for x in word.region])
            self.logger.debug(f"word alignment: {_str} [{_id} - {ifg}]")
            if i != 0:
                te = TextEquivType(
                    Unicode=_str, conf=conf, dataType="other", dataTypeDetails=f"ocrd-cis-word-alignment:{ddt}")
                words[0].region[0].add_TextEquiv(te)
            else:
                words[0].region[0].get_TextEquiv()[i].set_dataType("other")
                words[0].region[0].get_TextEquiv()[i].set_dataTypeDetails(f"ocrd-cis-word-alignment-master-ocr:{ddt}")
            words[0].region[0].get_TextEquiv()[i].set_index(i+1)

    def find_word(self, tokens, regions, t="other"):
        tokens_str = f"tokens = {tokens} [{t}]"
        self.logger.debug(tokens_str)
        for i, _ in enumerate(regions):
            n = self.match_tokens(tokens, regions, i)
            if n == 0:
                continue
            return tuple([regions[i:n], regions[i:]])
        # not found try again with levenshtein
        self.logger.warn(f"could not find {tokens_str}; trying again")
        for i, _ in enumerate(regions):
            n = self.match_tokens_lev(tokens, regions, i)
            if n == 0:
                continue
            return tuple([regions[i:n], regions[i:]])
        # not found try again to match token within another one
        self.logger.warn(f"could not find {tokens_str}; trying again")
        for i, _ in enumerate(regions):
            n = self.match_tokens_within(tokens, regions, i)
            if n == 0:
                continue
            return tuple([regions[i:n], regions[i:]])
        # nothing could be found
        return tuple([None, regions])

    def match_tokens(self, tokens, regions, i):
        def f(a, b):
            return a in b
            # if a and b:
            #     return a in b
            # return False
        return self.match_tokens_lambda(tokens, regions, i, f)

    def match_tokens_lev(self, tokens, regions, i):
        def f(a, b):
            k = 3  # int(len(a)/3)
            d = Levenshtein.distance(a, b)
            self.logger.debug(f"lev {a} <=> {b}: {d} ({d})")
            return d <= 1 or d <= k
        return self.match_tokens_lambda(tokens, regions, i, f)

    def match_tokens_within(self, tokens, regions, i):
        def f(a, b):
            return a in b
        return self.match_tokens_lambda(tokens, regions, i, f)

    def match_tokens_lambda(self, tokens, regions, i, f):
        """
        Returns one after the last index of the match starting from i.
        Returns 0 if nothing could be matched.
        """
        for j, token in enumerate(tokens):
            sum_i_j = j + i
            if sum_i_j >= len(regions):
                return 0
            unicode = regions[sum_i_j].TextEquiv[0].Unicode
            if not unicode:
                self.logger.warn(f"cannot find {token}")
                return 0
            self.logger.debug(f'checking {token} with {unicode}')
            if f(token, unicode):
                continue
            if j == 0:
                return 0
            # skip this and try next one
            # if we already have found a
            # match ath the first token position
            i += 1
        return i + len(tokens)

    def run_java_aligner(self, input_pcgts: List[OcrdPage]) -> str:
        lines = list()
        for pcgts in input_pcgts:
            lines.append([get_textequiv_unicode(line)
                          for line in pcgts.get_Page().get_AllTextLines()])
        # JavaAligner expects a strange input format
        lines = zip(*lines)
        _input = [x.strip() for t in lines for x in t]
        for i in _input:
            self.logger.debug("input line: %s", i)
        n = len(input_pcgts)
        self.logger.debug("starting java client")
        p = JavaAligner(n, getLevelName(self.logger.getEffectiveLevel()))
        return p.run("\n".join(_input))

class Alignment:
    file_grp: str
    pcgts: OcrdPage
    region: TextRegionType
    alignment: dict
    def __init__(self, file_grp: str, pcgts: OcrdPage, region: TextRegionType, alignment: dict):
        self.file_grp = file_grp
        self.pcgts = pcgts
        self.region = region
        self.alignment = alignment

def get_textequiv_unicode(r):
    if r is None or r.get_TextEquiv() is None or len(r.get_TextEquiv()) == 0:
        return ""
    return r.get_TextEquiv()[0].Unicode

def get_textequiv_conf(r):
    if r is None or r.get_TextEquiv() is None or len(r.get_TextEquiv()) == 0:
        return 0.0
    return r.get_TextEquiv()[0].conf
