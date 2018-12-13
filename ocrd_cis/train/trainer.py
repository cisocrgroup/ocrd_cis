from ocrd import Processor
from ocrd.utils import getLogger
from ocrd_cis import JavaTrain
from ocrd_cis import get_ocrd_tool
import ocrd_cis.train.config as config


class Trainer(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-train']
        kwargs['version'] = ocrd_tool['version']
        super(Trainer, self).__init__(*args, **kwargs)
        self.mpath = config.MPATH
        self.ppath = config.PPATH
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
