import subprocess
import logging
import json
from ocrd.utils import getLogger
from pathlib import Path

MAIN = "de.lmu.cis.ocrd.cli.Main"


def JavaAligner(jar, n):
    """Create a java process that calls -c align -D '{"n":n}'"""
    d = {'n': n}
    args = ['-c', 'align', '-D', "{}".format(json.dumps(d))]
    return JavaProcess(jar, args)


def JavaProfiler(jar, exe, backend, lang):
    d = {
        'executable': exe,
        'backend': backend,
        'language': lang,
    }
    args = ['-c', 'align', '-D', "{}".format(json.dumps(d))]
    return JavaProcess(jar, args)


class JavaProcess:
    def __init__(self, jar, args):
        self.jar = jar
        self.args = args
        self.main = MAIN
        self.log = getLogger('cis.JavaProcess')
        if not Path(jar).is_file():
            raise FileNotFoundError("no such file: {}".format(jar))

    def run(self, _input):
        """
        Run the process with the given input and get its output.
        The process writes _input to stdin of the process.
        """
        cmd = self.get_cmd()
        self.log.info('command: %s', " ".join(cmd))
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # only since 3.6: encoding='utf-8',
        ) as p:
            output, err = p.communicate(input=_input.encode('utf-8'))
            self.log_stderr(err)
            output = output.decode('utf-8')
            retval = p.wait()
            self.log.info("%s: %i", " ".join(cmd), retval)
            if retval != 0:
                raise ValueError(
                    "cannot execute {}: {}\nreturned: {}"
                    .format(cmd, err.decode('utf-8') if err else u'', retval))
            return output

    def log_stderr(self, err):
        for line in err.decode('utf-8').split("\n"):
            if "DEBUG:" in line:
                self.log.debug(line[6:])
            elif "INFO:" in line:
                self.log.info(line[5:])

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
