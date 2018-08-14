import subprocess
from ocrd.utils import getLogger
from pathlib import Path

ALLIGNER = "de.lmu.cis.ocrd.cli.Align"
PROFILER = "de.lmu.cis.ocrd.cli.Profile"


class JavaProcess:
    def __init__(self, jar, args):
        self.jar = jar
        self.args = args
        self.log = getLogger('JavaProcess')
        if not Path(jar).is_file():
            raise FileNotFoundError("no such file: {}".format(jar))

    def run_aligner(self, _input):
        self.run(ALLIGNER, _input)

    def run_profiler(self, _input):
        self.run(PROFILER, _input)

    def run(self, main, _input):
        """
        Run the given main-class of the jar.
        The process writes _input to stdin of the proces.
        """
        cmd = self.get_cmd(main)
        self.log.info('command: %s', cmd)
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                # only since 3.6: encoding='utf-8',
                # stderr=subprocess.DEVNULL,
        ) as p:
            self.output, err = p.communicate(input=_input.encode('utf-8'))
            self.output = self.output.decode('utf-8')
            retval = p.wait()
            self.log.info("%s: %i", cmd, retval)
            if retval != 0:
                raise ValueError(
                    "cannot execute {}: {}\nreturned: {}"
                    .format(cmd, err.decode('utf-8') if err else u'', retval))

    def get_cmd(self, main):
        cmd = ['java', '-cp', self.jar, main]
        cmd.extend(self.args)
        return cmd
