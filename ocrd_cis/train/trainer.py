from ocrd import Processor
import click

from ocrd.decorators import ocrd_cli_options
from ocrd.decorators import ocrd_cli_wrap_processor

from ocrd.utils import getLogger
from ocrd_cis import JavaTrain
from ocrd_cis import get_ocrd_tool

MPATH = ""
PPATH = ""


@click.command()
@ocrd_cli_options
def cis_ocrd_train(*args, **kwargs):
    global MPATH
    MPATH = kwargs["mets"]
    global PPATH
    PPATH = kwargs["parameter"]
    # kwargs["mpath"] = mpath
    # kwargs["ppath"] = ppath
    return ocrd_cli_wrap_processor(Trainer, *args, **kwargs)



class Trainer(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-train']
        kwargs['version'] = ocrd_tool['version']
        super(Trainer, self).__init__(*args, **kwargs)
        self.mpath = MPATH
        self.ppath = PPATH
        self.log = getLogger('cis.Processor.Trainer')

    def process(self):
        ifgs = self.input_file_grp.split(",")
        JavaTrain(
            jar=self.parameter["cisOcrdJar"],
            mets=self.mpath,
            parameter=self.ppath,
            ifgs=ifgs,
            loglvl="DEBUG",
        ).run("")
