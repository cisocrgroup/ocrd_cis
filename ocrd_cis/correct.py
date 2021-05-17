import os
import click
import json
from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_models.ocrd_mets import OcrdMets

from ocrd_cis.util import get_ocrd_tool, run_apoco


@click.command()
@ocrd_cli_options
def correct(*args, **kwargs):
    return ocrd_cli_wrap_processor(Corrector, *args, **kwargs)


class Corrector(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-post-correct']
        kwargs['version'] = ocrd_tool['version']
        super(Corrector, self).__init__(*args, **kwargs)
        self.mets = os.path.join(self.workspace.directory, "mets.xml")

    def process(self):
        run_apoco([
            "correct",
            "--log-level", "DEBUG",
            "-I", self.input_file_grp,
            "-O", self.output_file_grp,
            "-m", self.mets,
            "-p", json.dumps(self.parameter),
        ])
        # reload the mets file to prevent it from overriding the
        # updated version from the apoco process
        self.workspace.mets = OcrdMets(filename=self.mets)
