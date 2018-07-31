import json
from ocrd import Processor
from ocrd_cis import get_ocrd_tool
from ocrd.utils import getLogger
from ocrd.model.ocrd_page import from_file
from ocrd_cis import JavaProcess


class Profiler(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-profile']
        kwargs['version'] = ocrd_tool['version']
        super(Profiler, self).__init__(*args, **kwargs)
        self.log = getLogger('Processor.Profiler')

    def process(self):
        profile = self.read_profile()
        for k in profile:
            self.log.debug("k: %s", k)

    def read_profile(self):
        ifs = sorted(
            self.workspace.mets.find_files(fileGrp=self.input_file_grp),
            key=lambda ifile: ifile.ID
        )
        _input = []
        for ifile in ifs:
            pcgts = from_file(
                self.workspace.download_file(ifile)
            )
            for region in pcgts.get_Page().get_TextRegion():
                for line in region.get_TextLine():
                    self.log.debug("line: %s", line.get_TextEquiv()[0].Unicode)
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
        self.log.debug("JSON: %s", p.output)
        return json.loads(p.output)
