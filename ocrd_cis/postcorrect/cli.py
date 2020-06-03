from __future__ import absolute_import
import click
import json
import os
from ocrd import Processor
from ocrd.decorators import ocrd_cli_options
from ocrd.decorators import ocrd_cli_wrap_processor
from ocrd_utils import getLogger
from ocrd_models.ocrd_mets import OcrdMets
from ocrd_cis import JavaPostCorrector
from ocrd_cis import get_ocrd_tool

LOG_LEVEL = 'INFO'

@click.command()
@ocrd_cli_options
def ocrd_cis_postcorrect(*args, **kwargs):
    if 'log_level' in kwargs and kwargs['log_level']:
        global LOG_LEVEL
        LOG_LEVEL = kwargs['log_level']
    return ocrd_cli_wrap_processor(PostCorrector, *args, **kwargs)

class PostCorrector(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-postcorrect']
        kwargs['version'] = ocrd_tool['version']
        super(PostCorrector, self).__init__(*args, **kwargs)
        self.log = getLogger('cis.Processor.PostCorrector')

    def process(self):
        ifgs = self.input_file_grp.split(",")  # input file groups
        ofg = self.output_file_grp
        profiler = {}
        profiler["path"] = self.parameter["profilerPath"]
        profiler["config"] = self.parameter["profilerConfig"]
        profiler["noCache"] = True
        self.parameter["profiler"] = profiler
        self.parameter["runDM"] = True
        metspath = os.path.join(self.workspace.directory, "mets.xml")
        print(json.dumps(self.parameter, indent=4))
        p = JavaPostCorrector(metspath, ",".join(ifgs), ofg, self.parameter, LOG_LEVEL)
        p.exe()
        # reload the mets file to prevent it from overriding the
        # updated version from the java process
        self.workspace.mets = OcrdMets(filename=metspath)
