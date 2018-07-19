import subprocess
from ocrd.utils import getLogger

class JavaProcess:
    def __init__(self, jar, main, input, args):
        self.jar = jar
        self.main = main
        self.input = input
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
                #stderr=subprocess.DEVNULL,
        ) as p:
            # self.log.info("input: %s", self.get_input())
            self.output, err = p.communicate(input=self.get_input())
            # self.log.debug("data: %s", self.output)
            # for x in str(self.output).split("\n"):
            #     self.log.debug("x: %s", str(x))
            self.output = self.output.split("\n")
            retval = p.wait()
            self.log.info("retval: %i", retval)
            if retval != 0:
                raise ValueError(
                    "cannot execute {}: {}\nreturned: {}"
                    .format(cmd, err, retval))

    def get_input(self):
        return "\n".join(self.input)

    def get_cmd(self):
        cmd = ['java', '-cp', self.jar, self.main]
        cmd.extend(self.args)
        return cmd
