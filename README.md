# ocrd-cli

[CIS](http://www.cis.lmu.de) [OCR-D](http://ocr-d.de) command line tools

## General usage

### Virtualenv

Use `virtualenv` to install dependencies:
* `virtualenv -p python3.6 env-dir`
* `source env-dir/bin/activate`
* `pip install -r requirements.txt`
* `pip install -e path/to/dir/containing/setup.py`

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

## Tools

### ocrd-cis-align

The alignment tool line-aligns multiple file groups. It can be used to
align the results of multiple OCRs with their respective ground-truth.

The tool expects a comma-separated list of input file groups, the
according output file group and the url of the configuration file:

```sh
ocrd-cis-align \
  --input-file-grp 'ocr1,ocr2,gt' \
  --ouput-file-grp 'ocr1+ocr2+gt' \
  --parameter file:///path/to/config.json
```

## OCR-D links

- [OCR-D](https://ocr-d.github.io)
- [Github](https://github.com/OCR-D)
- [Project-page](http://www.ocr-d.de/)
- [Ground-truth](http://www.ocr-d.de/sites/all/GTDaten/IndexGT.html)
