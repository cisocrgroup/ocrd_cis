import subprocess
import json

from ocrd_utils import getLogger, resource_filename
from pathlib import Path

MAIN = "de.lmu.cis.ocrd.cli.Main"
JAR = str(resource_filename('ocrd_cis', 'data/ocrd-cis.jar'))

def JavaAligner(n, loglvl):
    """Create a java process that calls -c align -D '{"n":n}'"""
    return JavaProcess(JAR, ['-c', 'align',
                             '--log-level', loglvl,
                             '--parameter', '{}'.format(json.dumps({'n':n}))])

def JavaPostCorrector(mets, ifg, ofg, params, loglvl):
    return JavaProcess(JAR, ['-c', 'post-correct',
                             '--log-level', loglvl,
                             '--input-file-grp', ifg,
                             '--output-file-grp', ofg,
                             '--mets', mets,
                             '-p', "{}".format(json.dumps(params))])


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
        self.log.debug('input: %s', _input)
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
            self.log.debug("got output")
            retval = p.wait()
            self.log.debug("waited")
            self.log.debug("%s: %i", " ".join(cmd), retval)
            if retval != 0:
                raise ValueError(
                    "cannot execute {}: {}\n{}"
                    .format(" ".join(cmd), retval, err.decode('utf-8')))
            # self.log.info('output: %s', output)
            return output

    def exe(self):
        """
        Run the process with no imput returning no output.
        """
        cmd = self.get_cmd()
        self.log.info('command: %s', " ".join(cmd))
        with subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE
        ) as p:
            sout, eout = p.communicate()
            self.log_stderr(eout)
            retval = p.wait()
            if retval != 0:
                raise ValueError(
                    "cannot execute {}: {}\n{}"
                    .format(" ".join(cmd), retval, eout.decode('utf-8')))

    def log_stderr(self, err):
        for line in err.decode("utf-8").split("\n"):
            if line.startswith("DEBUG"):
                self.log.debug(line[6:])
            elif line.startswith("INFO"):
                self.log.info(line[5:])

    def get_cmd(self):
        cmd = ['java', '-Dfile.encoding=UTF-8',
               '-Xmx3g', '-cp', self.jar, self.main]
        cmd.extend(self.args)
        return cmd
