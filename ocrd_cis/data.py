import pkg_resources
import os
import sys

from ocrd_cis.util import apoco_exe


def get_path(args):
    usage = f'usage: {args[0]} -help|-exe|-config|-pre19th|-19th'
    if '-help' in args or '--help' in args:
        return usage
    if '-exe' in args or '--exe' in args:
        return apoco_exe()
    if '-config' in args or '--config' in args:
        return pkg_resources.resource_filename('ocrd_cis', os.path.join("data", "config.json"))
    if '-pre19th' in args or '--pre19th' in args:
        return pkg_resources.resource_filename('ocrd_cis', os.path.join("data", "pre19th.bin"))
    if '-19th' in args or '--19th' in args:
        return pkg_resources.resource_filename('ocrd_cis', os.path.join("data", "19th.bin"))
    raise ValueError(usage)


def data():
    print(get_path(sys.argv))


if __name__ == "__main__":
    data()
