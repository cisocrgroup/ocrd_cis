import subprocess
from ocrd.utils import getLogger
from pathlib import Path

ALLIGNER = "de.lmu.cis.ocrd.cli.Align"


class JavaProcess:
    def __init__(self, jar, args):
        self.jar = jar
        self.args = args
        self.log = getLogger('JavaProcess')
        if not Path(jar).is_file():
            raise FileNotFoundError("no such file: {}".format(jar))

    def run_aligner(self, _input):
        self.run(ALLIGNER, _input)

    def run(self, main, _input):
        cmd = self.get_cmd(main)
        self.log.info('command: %s', cmd)
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                encoding='utf-8',
                # stderr=subprocess.DEVNULL,
        ) as p:
            self.output, err = p.communicate(input=_input)
            self.output = self.output
            retval = p.wait()
            self.log.info("retval: %i", retval)
            if retval != 0:
                raise ValueError(
                    "cannot execute {}: {}\nreturned: {}"
                    .format(cmd, err, retval))

    def get_cmd(self, main):
        cmd = ['java', '-cp', self.jar, main]
        cmd.extend(self.args)
        return cmd
