import subprocess
import logging
from ocrd.utils import getLogger
from pathlib import Path

MAIN = "de.lmu.cis.ocrd.cli.Main"
ALIGNER = MAIN
PROFILER = "de.lmu.cis.ocrd.cli.Profile"


def JavaAligner(jar, args):
    """Create a java process that calls -c align"""
    args = ['-c', 'align'] + args
    return JavaProcess(jar, ALIGNER, args)


class JavaProcess:
    def __init__(self, jar, main, args):
        self.jar = jar
        self.args = args
        self.main = main
        self.log = getLogger('cis.JavaProcess')
        if not Path(jar).is_file():
            raise FileNotFoundError("no such file: {}".format(jar))

    def profiler(jar, args):
        return JavaProcess(jar, PROFILER, args)

    def run(self, _input):
        """
        Run the process with the given input and get its output.
        The process writes _input to stdin of the process.
        """
        cmd = self.get_cmd()
        self.log.info('command: %s', cmd)
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                # only since 3.6: encoding='utf-8',
                # stderr=subprocess.DEVNULL,
        ) as p:
            output, err = p.communicate(input=_input.encode('utf-8'))
            output = output.decode('utf-8')
            retval = p.wait()
            self.log.info("%s: %i", cmd, retval)
            if retval != 0:
                raise ValueError(
                    "cannot execute {}: {}\nreturned: {}"
                    .format(cmd, err.decode('utf-8') if err else u'', retval))
            return output

    def get_cmd(self):
        cmd = ['java', '-cp', self.jar, self.main]
        self.args.append('--log-level')
        self.args.append(self.get_log_level())
        cmd.extend(self.args)
        return cmd

    def get_log_level(self):
        level = logging.getLevelName(self.log.level)
        if level == 'NOTSET':
            level = 'INFO'
        return level
