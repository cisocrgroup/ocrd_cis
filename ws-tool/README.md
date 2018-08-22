
# Using tesserocr on folder with ground truth zips

[CIS](http://www.cis.lmu.de) [OCR-D](http://ocr-d.de) tool for unpacking and adding several ziped ground truth files to workspace

## General usage

### Essential OCR-D packages:
* ocrd core or cis version
* tesserocr

### Virtualenv
Don't forget to
`source env/bin/activate`
your virtualenv

### OCR-D workspace

add files to workspace:
```sh
python3.6 ocrd-wstool.py \
/path/to/workspace/ \
/path/to/folder containing gt-zips/
```


### Use Tesserocr on Workspace

```sh
ocrd-tesserocr-recognize \
--input-file-grp OCR-D-XML \
--output-file-grp OCR-D-REC \
--mets /path/to/workspace/mets.xml \
--parameter /path/to/parameter.json
```

If you want to use different Models,
name different output-file-groups and use the path to the specific parameter config file
