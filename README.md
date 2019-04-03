![build status](https://travis-ci.org/cisocrgroup/cis-ocrd-py.svg?branch=dev)
# cis-ocrd-py

[CIS](http://www.cis.lmu.de) [OCR-D](http://ocr-d.de) command line tools

## General usage

### Essential system packages

```sh
sudo apt-get install \
  git \
  build-essential \
  python3 python3-pip \
  libxml2-dev \
  default-jdk
```



### Virtualenv

Use `virtualenv` to install dependencies:
* `virtualenv -p python3.6 env`
* `source env/bin/activate`
* `pip install -e path/to/dir/containing/setup.py`

Use `deactivate` to deactivate the virtualenv again.

### OCR-D workspace

* Create a new (empty) workspace: `ocrd workspace init workspace-dir`
* cd into `workspace-dir`
* Add new file to workspace: `ocrd workspace add file -G group -i id
  -m mimetype`

### Tests

Issue `make test` to run the automated test suite. The tests depend on
the following tools:

* [wget](https://www.gnu.org/software/wget/)
* [envsubst](https://linux.die.net/man/1/envsubst)

You can run individual testcases using the `run_*_test.bash` scripts in
the tests directory. Use the `--persistent` or `-p` flag to keep
temporary directories.

You can override the temporary directory by setting the `TMP_DIR` environment
variable.

## Tools

### ocrd-cis-align

The alignment tool line-aligns multiple file groups. It can be used to
align the results of multiple OCRs with their respective ground-truth.

The tool expects a comma-separated list of input file groups, the
according output file group and the url of the configuration file:

```sh
ocrd-cis-align \
  --input-file-grp 'ocr1,ocr2,gt' \
  --output-file-grp 'ocr1+ocr2+gt' \
  --mets mets.xml \
  --parameter file:///path/to/config.json
```


### ocrd-cis-ocropy-train
The ocropy-train tool can be used to train lstm models.
The tool takes the ground truth from a workspace and safes the snippets from the corresponding page.
Then the model is trained on all snippets for 1 million randomized iterations or the given number from the parameter file.
```sh
ocrd-cis-ocropy-train \
  --input-file-grp 'OCR-D-XML' \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

### ocrd-cis-ocropy-recognize
The ocropy-recognize tool can be used to recognize lines / words / glyphs from pages of a workspace.
The tool runs the ocropy optical character recognition for each "region" given in the XML file of the workspace.
```sh
ocrd-cis-ocropy-recognize \
  --input-file-grp 'OCR-D-XML' \
  --output-file-grp 'OCR-D-OCROPY' \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

## All in One Tool
For the all in One Tool install all above tools and Tesserocr as explained below.
Then use it like:
```sh
ocrd-cis-aio --parameter file:///path/to/config.json
```


### Tesserocr
Install essential system packages for Tesserocr
```sh
sudo apt-get install python3-tk \
  tesseract-ocr libtesseract-dev libleptonica-dev \
  libimage-exiftool-perl libxml2-utils
```

Then install Tesserocr from: https://github.com/OCR-D/ocrd_tesserocr
```sh
pip install -r requirements.txt
pip install .
```

Download and move tesseract models from:
https://github.com/tesseract-ocr/tesseract/wiki/Data-Files
or use your own models
place them into: /usr/share/tesseract-ocr/4.00/tessdata

Tesserocr v2.4.0 seems broken for tesseract 4.0.0-beta. Install
Version v2.3.1 instead: `pip install tesseract==2.3.1`.



## OCR-D links

- [OCR-D](https://ocr-d.github.io)
- [Github](https://github.com/OCR-D)
- [Project-page](http://www.ocr-d.de/)
- [Ground-truth](http://www.ocr-d.de/sites/all/GTDaten/IndexGT.html)
