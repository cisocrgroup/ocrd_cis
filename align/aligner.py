from __future__ import absolute_import
from ocrd import Processor
from ocrd.utils import getLogger
from ocrd.model.ocrd_page import from_file
from lib.javaprocess import JavaProcess
from align.ocrd_tool import get_ocrd_tool
#from ocrd import MIME_TYPE


class Aligner(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['cis-ocrd-align']
        kwargs['version'] = ocrd_tool['version']
        super(Aligner, self).__init__(*args, **kwargs)
        self.log = getLogger('Processor.Aligner')

    def process(self):
        lines = self.zip_lines(self.input_file_grp.split(","))
        output = self.run_ocrd_aligner(lines)
        for o in output.split("\n"):
            self.log.info("o: %s", o)

    def run_ocrd_aligner(self, lines):
        """Run the external java aligner over the zipped lines"""
        if not lines:
            return
        n = len(lines[0])
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
