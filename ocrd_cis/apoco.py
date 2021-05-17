import sys
import subprocess
from pathlib import Path

from ocrd_cis.util import run_apoco
from ocrd_cis.util import apoco_exe


def apoco():
    exe = apoco_exe()
    if not Path(exe).is_file():
        raise FileNotFoundError(f"no such file: {exe}")
    cmd = [exe]
    cmd.extend(sys.argv[1:])
    with subprocess.Popen(cmd, stderr=subprocess.PIPE) as p:
        stdout, stderr = p.communicate()
        if stderr:
            for line in stderr.decode("utf-8").split("\n"):
                print(line, file=sys.stderr)
        if stdout:
            for line in stdout.decode("utf-8").split("\n"):
                print(line)
        ret = p.wait()
        if ret != 0:
            raise ValueError(f"error executing {' '.join(cmd)}: {ret}")


if __name__ == "__main__":
    apoco()
