import pkg_resources
import sys

def main():
    usage = 'usage: ' + sys.argv[0] + ' -jar|-3gs|-model|-config'
    if '-h' in sys.argv:
        print(usage)
    elif '-jar' in sys.argv:
        print(pkg_resources.resource_filename('ocrd_cis', 'data/ocrd-cis.jar'))
    elif '-3gs' in sys.argv:
        print(pkg_resources.resource_filename('ocrd_cis', 'data/3gs.csv.gz'))
    elif '-model' in sys.argv:
        print(pkg_resources.resource_filename('ocrd_cis', 'data/model.zip'))
    elif '-config' in sys.argv:
        print(pkg_resources.resource_filename('ocrd_cis', 'data/config.json'))
    else:
        raise ValueError(usage)

if __name__ == "__main__":
    main()
