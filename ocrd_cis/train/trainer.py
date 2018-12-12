from ocrd_cis import JavaProfiler
from ocrd_cis import get_ocrd_tool


class Trainer(Processor):
    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-trainer']
        kwargs['version'] = ocrd_tool['version']
        super(Trainer, self).__init__(*args, **kwargs)
        self.log = getLogger('cis.Processor.Trainer')
