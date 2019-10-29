from ocrd import Processor
from ocrd.decorators import ocrd_cli_options
from ocrd.decorators import ocrd_cli_wrap_processor
from ocrd_cis import get_ocrd_tool
from ocrd_modelfactory import page_from_file
from ocrd_utils import getLogger
import click
import json
import os

@click.command()
@ocrd_cli_options
def cis_ocrd_wer(*args, **kwargs):
    return ocrd_cli_wrap_processor(WERer, *args, **kwargs)

class WERer(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-wer']
        kwargs['version'] = ocrd_tool['version']
        super(WERer, self).__init__(*args, **kwargs)
        self.log = getLogger('cis.Processor.WERer')
        self.testi = self.parameter['testIndex']
        self.gti = self.parameter['gtIndex']

    def process(self):
        """Calculate word error rate"""
        stats = Stats()
        for (n, input_file) in enumerate(self.input_files):
            self.log.info("INPUT FILE %i / %s", n,
                          input_file.pageId or input_file.ID)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            for region in pcgts.get_Page().get_TextRegion():
                for line in region.get_TextLine():
                    for word in line.get_Word():
                        test = word.get_TextEquiv()[self.testi]
                        gt = word.get_TextEquiv()[self.gti]
                        stats.add(test, gt)
        stats.calculate()
        base = self.output_file_grp
        out = self.workspace.add_file(
            ID=base,
            file_grp=self.output_file_grp,
            local_filename=os.path.join(self.output_file_grp, base + ".json"),
            mimetype="application/json",
            content=json.dumps(stats.__dict__))
        self.log.info("created file ID: %s, file_grp: %s, path: %s",
                      base, self.output_file_grp, out.local_filename)

class Stats:
    def __init__(self):
        self.totalWords = 0
        self.correctWords = 0
        self.incorrectWords = 0

    def add(self, test, gt):
        self.totalWords += 1
        if test == gt:
            self.correctWords += 1
        else:
            self.incorrectWords += 1

    def calculate(self):
        self.wordErrorRate = 0
        if self.totalWords == 0:
            return
        self.wordErrorRate = self.incorrectWords / self.totalWords
