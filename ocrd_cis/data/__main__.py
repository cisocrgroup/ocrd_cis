import pkg_resources
import sys

def main():
    if '-jar' in sys.argv:
        print(pkg_resources.resource_filename('ocrd_cis', 'data/ocrd-cis.jar'))
    elif '-3gs' in sys.argv:
        print(pkg_resources.resource_filename('ocrd_cis', 'data/3gs.csv.gz'))
    elif '-model' in sys.argv:
        print(pkg_resources.resource_filename('ocrd_cis', 'data/model.zip'))
    else:
        raise ValueError('usage: ' + sys.argv[0] + ' -jar|-3gs|-model')

if __name__ == "__main__":
    main()
