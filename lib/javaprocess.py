import subprocess
from ocrd.utils import getLogger

class JavaProcess:
    def __init__(self, jar, main, input_str, args):
        self.jar = jar
        self.main = main
        self.input_str = input_str
        self.args = args
        self.log = getLogger('JavaProcess')

    def run(self):
        cmd = self.get_cmd()
        self.log.info('command: %s', cmd)
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                encoding='utf-8',
                stderr=subprocess.DEVNULL,
        ) as p:
            self.output, err = p.communicate(input=self.input_str)
            self.output = self.output
            retval = p.wait()
            self.log.info("retval: %i", retval)
            if retval != 0:
                raise ValueError(
                    "cannot execute {}: {}\nreturned: {}"
                    .format(cmd, err, retval))

    def get_cmd(self):
        cmd = ['java', '-cp', self.jar, self.main]
        cmd.extend(self.args)
        return cmd
