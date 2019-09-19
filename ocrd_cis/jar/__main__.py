import pkg_resources

def main(args=None):
    print(pkg_resources.resource_filename('ocrd_cis', 'jar/ocrd-cis.jar'))

if __name__ == "__main__":
    main()
