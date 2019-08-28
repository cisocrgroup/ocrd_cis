import click

from ocrd.decorators import ocrd_cli_options
from ocrd.decorators import ocrd_cli_wrap_processor
from ocrd import Processor
from ocrd_utils import getLogger
from ocrd_cis import get_ocrd_tool
from ocrd_cis import JavaProfiler

@click.command()
@ocrd_cli_options
def cis_ocrd_profile(*args, **kwargs):
    return ocrd_cli_wrap_processor(Profiler, *args, **kwargs)

class Profiler(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-profile']
        kwargs['version'] = ocrd_tool['version']
        super(Profiler, self).__init__(*args, **kwargs)
        self.log = getLogger('cis.Processor.Profiler')

    def process(self):
        self.log.debug("starting java client")
        p = JavaProfiler(self.workspace.mets_target, self.input_file_grp,
                         self.output_file_grp, self.parameter, "DEBUG")
        p.exe()
        # Reload the updated METS file to make sure that run_processor
        # does not overwrite the updated file with the old.
        self.workspace.reload_mets()
