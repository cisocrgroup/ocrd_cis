from __future__ import absolute_import
import click
import json
import os
from ocrd import Processor
from ocrd.decorators import ocrd_cli_options
from ocrd.decorators import ocrd_cli_wrap_processor
from ocrd_utils import getLogger, getLevelName
from ocrd_models.ocrd_mets import OcrdMets
from ocrd_cis import JavaPostCorrector
from ocrd_cis import get_ocrd_tool

@click.command()
@ocrd_cli_options
def ocrd_cis_postcorrect(*args, **kwargs):
    return ocrd_cli_wrap_processor(PostCorrector, *args, **kwargs)

class PostCorrector(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-postcorrect']
        kwargs['version'] = ocrd_tool['version']
        super(PostCorrector, self).__init__(*args, **kwargs)

    def process(self):
        self.log = getLogger('processor.CISPostCorrector')
        profiler = {}
        profiler["path"] = self.parameter["profilerPath"]
        profiler["config"] = self.parameter["profilerConfig"]
        profiler["noCache"] = True
        self.parameter["profiler"] = profiler
        self.parameter["runDM"] = True
        self.log.debug(json.dumps(self.parameter, indent=4))
        p = JavaPostCorrector(self.workspace.mets_target,
                              self.input_file_grp,
                              self.output_file_grp,
                              self.parameter,
                              getLevelName(self.log.getEffectiveLevel()))
        p.exe()
        # reload the mets file to prevent run_processor's save_mets
        # from overriding the results from the Java process
        self.workspace.reload_mets()
        # workaround for cisocrgroup/ocrd-postcorrection#13 (absolute paths in output):
        for output_file in self.workspace.find_files(file_grp=self.output_file_grp):
            flocat = output_file._el.find('{http://www.loc.gov/METS/}FLocat')
            flocat.attrib['LOCTYPE'] = 'OTHER'
            flocat.attrib['OTHERLOCTYPE'] = 'FILE'
            output_file.local_filename = os.path.relpath(output_file.local_filename, self.workspace.directory)
