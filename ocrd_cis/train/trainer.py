from ocrd import Processor
from ocrd.utils import getLogger
from ocrd_cis import JavaTrain
from ocrd_cis import get_ocrd_tool
import ocrd_cis.train.config as config
import os.path
import errno


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
        self.setup_model_dirs()
        ifgs = self.input_file_grp.split(",")
        JavaTrain(
            jar=self.parameter["jar"],
            mets=self.mpath,
            parameter=self.ppath,
            ifgs=ifgs,
            loglvl=config.LOG_LEVEL,
        ).run("")

    def setup_model_dirs(self):
        self.mk_model_dir(self.parameter["dleTraining"]["dynamicLexicon"])
        self.mk_model_dir(self.parameter["dleTraining"]["model"])
        self.mk_model_dir(self.parameter["dleTraining"]["training"])

    def mk_model_dir(self, filepath):
        bdir = os.path.dirname(filepath)
        self.log.debug("creating dir: %s", bdir)
        try:
            os.makedirs(bdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
