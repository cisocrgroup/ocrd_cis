from ocrd import Processor
from ocrd_cis import get_ocrd_tool
from ocrd.utils import getLogger
from ocrd.model.ocrd_page import from_file


class Profiler(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-align']
        kwargs['version'] = ocrd_tool['version']
        super(Profiler, self).__init__(*args, **kwargs)
        self.log = getLogger('Processor.Profiler')

    def process(self):
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
                    self.log.info("line: %s", line.get_TextEquiv()[0].Unicode)
        pass
        # p = JavaProcess(
        #     jar=self.parameter['cisOcrdJar'],
        #     args=[str(n)]
        # )
