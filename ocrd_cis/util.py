import json
import os
import subprocess
import sys
import pkg_resources
import re
from pathlib import Path

from ocrd_utils import getLogger
from pkg_resources import resource_string


def apoco_name():
    if sys.platform == "win32":
        return "apoco.exe"
    return "apoco." + sys.platform


def apoco_exe():
    return pkg_resources.resource_filename('ocrd_cis', os.path.join("data", apoco_name()))


def get_ocrd_tool():
    return json.loads(
        resource_string(__name__, 'ocrd-tool.json').decode('utf8'))


def run_apoco(args):
    """Run the apoco post-correction with the given args."""
    exe = apoco_exe()
    if not Path(exe).is_file():
        raise FileNotFoundError(f"no such file: {exe}")
    logger = getLogger('ocrd_cis')
    cmd = [exe]
    cmd.extend(args)
    logger.debug(f"running command: {' '.join(cmd)}")
    with subprocess.Popen(cmd, stderr=subprocess.PIPE) as p:
        _, std_err = p.communicate()
        log_stderr(logger, std_err)
        ret = p.wait()
        if ret != 0:
            raise ValueError(f"error executing {' '.join(cmd)}: {ret}")


def log_stderr(logger, err):
    if err:
        # match the YYYY/MM/DD HH:MM:SS prefix.
        p = re.compile(r'^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+')
        for line in err.decode("utf-8").split("\n"):
            if p.match(line):
                logger.debug(line[20:])
            elif len(line) > 0:
                logger.debug(line)
