import os
import json
import click
from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_models.ocrd_mets import OcrdMets

from ocrd_cis.util import get_ocrd_tool, run_apoco


@click.command()
@ocrd_cli_options
def align(*args, **kwargs):
    return ocrd_cli_wrap_processor(Aligner, *args, **kwargs)


class Aligner(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-align']
        kwargs['version'] = ocrd_tool['version']
        super(Aligner, self).__init__(*args, **kwargs)
        self.mets = os.path.join(self.workspace.directory, "mets.xml")

    def process(self):
        run_apoco(
            [
                "align",
                "--log-level", "DEBUG",
                "-I", self.input_file_grp,
                "-O", self.output_file_grp,
                "-m", self.mets,
            ]
        )
        # reload the mets file to prevent it from overriding the
        # updated version from the apoco process
        self.workspace.mets = OcrdMets(filename=self.mets)
