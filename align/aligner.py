from __future__ import absolute_import
import json
from ocrd import Processor
from ocrd.utils import getLogger
from ocrd.model.ocrd_page import from_file
from lib.javaprocess import JavaProcess
from align.ocrd_tool import get_ocrd_tool


class Aligner(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['cis-ocrd-align']
        kwargs['version'] = ocrd_tool['version']
        super(Aligner, self).__init__(*args, **kwargs)
        self.log = getLogger('Processor.Aligner')

    def process(self):
        ifgs = self.input_file_grp.split(",")  # input file groups
        ifts = self.zip_input_files(ifgs)  # input file tuples
        page_alignments = list()
        for ift in ifts:
            page_alignments.append(
                PageAlignment(
                    self,
                    # self.workspace,
                    # self.parameter['cisOcrdJar'],
                    ifgs,
                    ift))
        for pa in page_alignments:
            for la in pa.line_alignments:
                self.log.info("%s", la)

    def zip_input_files(self, ifgs):
        """Zip files of the given input file groups"""
        files = list()
        for ifg in ifgs:
            self.log.info("input file group: %s", ifg)
            ifiles = sorted(
                self.workspace.mets.find_files(fileGrp=ifg),
                key=lambda ifile: ifile.ID)
            for i in ifiles:
                self.log.info("sorted file: %s %s", i.url, i.ID)
            self.log.info("input files: %s", ifiles)
            files.append(ifiles)
        return zip(*files)


class PageAlignment:
    """PageAlignment holds a list of LineAlignments."""
    def __init__(self, process, ifgs, ifs):
        """Create a page alignment form a list of input files."""
        self.process = process
        # self.jar = jar
        self.ifgs = ifgs
        self.ifs = ifs
        self.log = getLogger('PageAlignment')
        self.align_lines()

    def align_lines(self):
        lines = list()
        for ifile in self.ifs:
            lines.append(self.read_lines_from_input_file(ifile))
        lines = zip(*lines)
        _input = [x for t in lines for x in t]
        for i in _input:
            self.log.info("input line: %s", i)
        n = len(self.ifs)
        p = JavaProcess(
            jar=self.process.parameter['cisOcrdJar'],
            main="de.lmu.cis.ocrd.cli.Align",
            input_str="\n".join(_input),
            args=[str(n)])
        p.run()
        lines = p.output.split("\n")
        self.line_alignments = list()
        for i in range(0, len(lines), n):
            self.line_alignments.append(LineAlignment(lines[i:i+n]))

    def read_lines_from_input_file(self, ifile):
        self.log.debug("reading input file: %s", ifile.url)
        lines = list()
        pcgts = from_file(self.process.workspace.download_file(ifile))
        for region in pcgts.get_Page().get_TextRegion():
            for line in region.get_TextLine():
                lines.append(line.get_TextEquiv()[0].Unicode)
        return lines


class LineAlignment:
    """
    LineAlignment holds a line alignment.
    A line alignment of n lines holds n-1 pairwise alignments
    and a list of token alignments of n-tuples.

    Each pairwise alignment represents the alignment of the
    master line with another. Pairwise aligned lines have always
    the same length. Underscores ('_') mark deletions or insertions.
    """
    def __init__(self, lines):
        """
        Create a LineAlignment from n-1 pairwise
        alignments an one token alignment at pos n-1.
        """
        self.n = len(lines)
        self.pairwise = list()
        for i in range(0, self.n-1):
            self.pairwise.append(tuple(lines[i].split(",")))
        self.tokens = list()
        for ts in lines[self.n-1].split(","):
            self.tokens.append(tuple(ts.split(":")))

    def __str__(self):
        data = {}
        data['pairwise'] = self.pairwise
        data['tokens'] = self.tokens
        return json.dumps(data)
