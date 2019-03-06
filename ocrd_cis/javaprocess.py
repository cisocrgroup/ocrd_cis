import subprocess
import json
from ocrd_utils import getLogger
from pathlib import Path

MAIN = "de.lmu.cis.ocrd.cli.Main"


def JavaAligner(jar, n, loglvl="DEBUG"):
    """Create a java process that calls -c align -D '{"n":n}'"""
    d = {'n': n}
    args = [
        '-c', 'align',
        "--log-level", loglvl,
        '-D', "{}".format(json.dumps(d))
    ]
    return JavaProcess(jar, args)


def JavaProfiler(jar, exe, backend, lang,
                 args, loglvl="DEBUG"):
    d = {
        'executable': exe,
        'backend': backend,
        'language': lang,
        'args': args,
    }
    args = [
        '-c', 'profile',
        "--log-level", loglvl,
        '-D', "{}".format(json.dumps(d))
    ]
    return JavaProcess(jar, args)


def JavaTrain(jar, mets, ifgs, parameter, loglvl="DEBUG"):
    args = [
        "-c", "train",
        "--mets", mets,
        "--log-level", loglvl,
        "--parameter", parameter
    ]
    for ifg in ifgs:
        args.append("-I")
        args.append(ifg)
    return JavaProcess(jar, args)


def JavaEvalDLE(jar, mets, ifgs, parameter, loglvl="DEBUG"):
    args = [
        '-c', 'evaluate-dle',
        '--mets', mets,
        '--log-level', loglvl,
        '--parameter', parameter
    ]
    for ifg in ifgs:
        args.append('-I')
        args.append(ifg)
    return JavaProcess(jar, args)


def JavaEvalRRDM(jar, mets, ifgs, parameter, loglvl="DEBUG"):
    args = [
        '-c', 'evaluate-rrdm',
        '--mets', mets,
        '--log-level', loglvl,
        '--parameter', parameter
    ]
    for ifg in ifgs:
        args.append('-I')
        args.append(ifg)
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
        self.log.info('input: %s', _input)
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
                    "cannot execute {}: {}\n{}"
                    .format(" ".join(cmd), retval, err.decode('utf-8')))
            # self.log.info('output: %s', output)
            return output

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
