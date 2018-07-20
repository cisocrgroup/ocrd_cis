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
        lines = self.zip_lines(self.input_file_grp.split(","))
        if not lines:
            return
        n = len(lines[0])
        output = self.run_ocrd_aligner(lines, n)
        alignments = self.get_alignments(output, n)
        for a in alignments:
            self.log.info("alignment: %s", a)

    def get_alignments(self, output, n):
        lines = output.split("\n")
        alignments = list()
        for i in range(0, len(lines), n):
            alignments.append(LineAlignment(lines[i:i+n]))
        return alignments

    def run_ocrd_aligner(self, lines, n):
        """Run the external java aligner over the zipped lines"""
        _input = [x for t in lines for x in t]
        p = JavaProcess(
            jar=self.parameter['cisOcrdJar'],
            main="de.lmu.cis.ocrd.cli.Align",
            input_str="\n".join(_input),
            args=[str(n)])
        p.run()
        return p.output

    def zip_lines(self, ifgs):
        """
        Read lines from input-file-groups.
        Returns a list of the line-aligned tuples
        """
        lines = list()
        for ifg in ifgs:
            self.log.info("input file group: %s", ifg)
            lines.append(self.read_lines(ifg))
        return list(zip(*lines))

    def read_lines(self, ifg):
        """Read all lines from an input-file-group (sorted by ID)"""
        ifiles = sorted(
            self.workspace.mets.find_files(fileGrp=ifg),
            key=lambda ifile: ifile.ID)
        lines = list()
        for ifile in ifiles:
            self.log.info("input file: %s", ifile)
            pcgts = from_file(self.workspace.download_file(ifile))
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
