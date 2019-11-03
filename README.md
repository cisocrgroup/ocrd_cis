![build status](https://travis-ci.org/cisocrgroup/ocrd_cis.svg?branch=dev)
# ocrd_cis

[CIS](http://www.cis.lmu.de) [OCR-D](http://ocr-d.de) command line
tools for the automatic post-correction of OCR-results.

## Introduction
`ocrd_cis` contains different tools for the automatic post correction
of OCR-results.  It contains tools for the training, evaluation and
execution of the post correction.  Most of the tools are following the
[OCR-D cli conventions](https://ocr-d.github.io/cli).

There is a helper tool to align multiple OCR results as well as a
version of ocropy that works with python3.

## Installation
There are multiple ways to install the `ocrd_cis` tools:
 * `make install` uses `pip` to install `ocrd_cis` (see below).
 * `make install-devel` uses `pip -e` to install `ocrd_cis` (see
   below).
 * `pip install --upgrade pip ocrd_cis_dir`
 * `pip install -e --upgrade pip ocrd_cis_dir`

It is possible to install `ocrd_cis` in a custom directory using
`virtualenv`:
```sh
 python3 -m venv venv-dir
 source venv-dir/bin/activate
 make install # or any other command to install ocrd_cis (see above)
 # use ocrd_cis
 deactivate
```

## Usage
Most tools follow the [OCR-D cli
conventions](https://ocr-d.github.io/cli).  They accept the
`--input-file-grp`, `--output-file-grp`, `--parameter`, `--mets`,
`--log-level` command line arguments (short and long).  For some tools
(most notably the alignment tool) expect a comma seperated list of
multiple input file groups.

The [ocrd-tool.json](ocrd_cis/ocrd-tool.json) contains a schema
description of the parameter config file for the different tools that
accept the `--parameter` argument.

### ocrd-cis-post-correct.sh
This bash script runs the post correction using a pre-trained
[model](http://cis.lmu.de/~finkf/model.zip).  If additional support
OCRs should be used, models for these OCR steps are required and must
be configured in an according configuration file (see ocrd-tool.json).

Arguments:
 * `--parameter` path to configuration file
 * `--input-file-grp` name of the master-OCR file group
 * `--output-file-grp` name of the post-correction file group
 * `--log-level` set log level
 * `--mets` path to METS file in workspace

### ocrd-cis-align
Aligns tokens of multiple input file groups to one output file group.
This tool is used to align the master OCR with any additional support
OCRs.  It accepts a comma-separated list of input file groups, which
it aligns in order.

Arguments:
 * `--parameter` path to configuration file
 * `--input-file-grp` comma seperated list of the input file groups;
   first input file group is the master OCR
 * `--output-file-grp` name of the file group for the aligned result
 * `--log-level` set log level
 * `--mets` path to METS file in workspace

### ocrd-cis-train.sh
Script to train a model from a list of ground-truth archives (see
ocrd-tool.json) for the post correction.  The tool somewhat mimics the
behaviour of other ocrd tools:
 * `--mets` for the workspace
 * `--log-level` is passed to other tools
 * `--parameter` is used as configuration
 * `--output-file-grp` defines the output file group for the model

### ocrd-cis-data
Helper tool to get the path of the installed data files. Usage:
`ocrd-cis-data [-jar|-3gs]` to get the path of the jar library or the
path to th default 3-grams language model file.

### ocrd-cis-wer
Helper tool to calculate the word error rate aligned ocr files.  It
writes a simple JSON-formated stats file to the given output file group.

Arguments:
 * `--input-file-grp` input file group of aligned ocr results with
   their respective ground truth.
 * `--output-file-grp` name of the file group for the stats file
 * `--log-level` set log level
 * `--mets` path to METS file in workspace

### ocrd-cis-profile
Run the profiler over the given files of the according the given input
file grp and adds a gzipped JSON-formatted profile to the output file
group of the workspace.  This tools requires an installed [language
profiler](https://github.com/cisocrgroup/Profiler).

Arguments:
 * `--parameter` path to configuration file
 * `--input-file-grp` name of the input file group to profile
 * `--output-file-grp` name of the output file group where the profile
   is stored
 * `--log-level` set log level
 * `--mets` path to METS file in the workspace

### ocrd-cis-ocropy-train
The ocropy-train tool can be used to train LSTM models.
It takes ground truth from the workspace and saves (image+text) snippets from the corresponding pages.
Then a model is trained on all snippets for 1 million (or the given number of) randomized iterations from the parameter file.
```sh
ocrd-cis-ocropy-train \
  --input-file-grp OCR-D-GT-SEG-LINE \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

### ocrd-cis-ocropy-clip
The ocropy-clip tool can be used to remove intrusions of neighbouring segments in regions / lines of a workspace.
It runs a (ad-hoc binarization and) connected component analysis on every text region / line of every PAGE in the input file group, as well as its overlapping neighbours, and for each binary object of conflict, determines whether it belongs to the neighbour, and can therefore be clipped to white. It references the resulting segment image files in the output PAGE (as AlternativeImage).
```sh
ocrd-cis-ocropy-clip \
  --input-file-grp OCR-D-SEG-LINE \
  --output-file-grp OCR-D-SEG-LINE-CLIP \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

### ocrd-cis-ocropy-resegment
The ocropy-resegment tool can be used to remove overlap between lines of a workspace.
It runs a (ad-hoc binarization and) line segmentation on every text region of every PAGE in the input file group, and for each line already annotated, determines the label of largest extent within the original coordinates (polygon outline) in that line, and annotates the resulting coordinates in the output PAGE.
```sh
ocrd-cis-ocropy-resegment \
  --input-file-grp OCR-D-SEG-LINE \
  --output-file-grp OCR-D-SEG-LINE-RES \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

### ocrd-cis-ocropy-segment
The ocropy-segment tool can be used to segment regions into lines.
It runs a (ad-hoc binarization and) line segmentation on every text region of every PAGE in the input file group, and adds a TextLine element with the resulting polygon outline to the annotation of the output PAGE.
```sh
ocrd-cis-ocropy-segment \
  --input-file-grp OCR-D-SEG-BLOCK \
  --output-file-grp OCR-D-SEG-LINE \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

### ocrd-cis-ocropy-deskew
The ocropy-deskew tool can be used to deskew pages / regions of a workspace.
It runs the Ocropy thresholding and deskewing estimation on every segment of every PAGE in the input file group and annotates the orientation angle in the output PAGE.
```sh
ocrd-cis-ocropy-deskew \
  --input-file-grp OCR-D-SEG-LINE \
  --output-file-grp OCR-D-SEG-LINE-DES \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

### ocrd-cis-ocropy-denoise
The ocropy-denoise tool can be used to despeckle pages / regions / lines of a workspace.
It runs the Ocropy "nlbin" denoising on every segment of every PAGE in the input file group and references the resulting segment image files in the output PAGE (as AlternativeImage).
```sh
ocrd-cis-ocropy-denoise \
  --input-file-grp OCR-D-SEG-LINE-DES \
  --output-file-grp OCR-D-SEG-LINE-DEN \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

### ocrd-cis-ocropy-binarize
The ocropy-binarize tool can be used to binarize, denoise and deskew pages / regions / lines of a workspace.
It runs the Ocropy "nlbin" adaptive thresholding, deskewing estimation and denoising on every segment of every PAGE in the input file group and references the resulting segment image files in the output PAGE (as AlternativeImage). (If a deskewing angle has already been annotated in a region, the tool respects that and rotates accordingly.) Images can also be produced grayscale-normalized.
```sh
ocrd-cis-ocropy-binarize \
  --input-file-grp OCR-D-SEG-LINE-DES \
  --output-file-grp OCR-D-SEG-LINE-BIN \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

### ocrd-cis-ocropy-dewarp
The ocropy-dewarp tool can be used to dewarp text lines of a workspace.
It runs the Ocropy baseline estimation and dewarping on every line in every text region of every PAGE in the input file group and references the resulting line image files in the output PAGE (as AlternativeImage).
```sh
ocrd-cis-ocropy-dewarp \
  --input-file-grp OCR-D-SEG-LINE-BIN \
  --output-file-grp OCR-D-SEG-LINE-DEW \
  --mets mets.xml
  --parameter file:///path/to/config.json
```

### ocrd-cis-ocropy-recognize
The ocropy-recognize tool can be used to recognize lines / words / glyphs from pages of a workspace.
It runs the Ocropy optical character recognition on every line in every text region of every PAGE in the input file group and adds the resulting text annotation in the output PAGE.
```sh
ocrd-cis-ocropy-recognize \
  --input-file-grp OCR-D-SEG-LINE-DEW \
  --output-file-grp OCR-D-OCR-OCRO \
  --mets mets.xml
  --parameter file:///path/to/config.json
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
or use your own models and
place them into: /usr/share/tesseract-ocr/4.00/tessdata

## Workflow configuration

A decent pipeline might look like this:

1. page-level cropping
2. page-level binarization
3. page-level deskewing
4. page-level dewarping
5. region segmentation
6. region-level clipping
7. region-level deskewing
8. line segmentation
9. line-level clipping or resegmentation
10. line-level dewarping
11. line-level recognition
12. line-level alignment

If GT is used, steps 1, 5 and 8 can be omitted. Else if a segmentation is used in 5 and 8 which does not produce overlapping sections, steps 6 and 9 can be omitted.

## Testing
To run a few basic tests type `make test` (`ocrd_cis` has to be
installed in order to run any tests).

## OCR-D workspace

* Create a new (empty) workspace: `ocrd workspace init workspace-dir`
* cd into `workspace-dir`
* Add new file to workspace: `ocrd workspace add file -G group -i id
  -m mimetype`

## OCR-D links

- [OCR-D](https://ocr-d.github.io)
- [Github](https://github.com/OCR-D)
- [Project-page](http://www.ocr-d.de/)
- [Ground-truth](http://www.ocr-d.de/sites/all/GTDaten/IndexGT.html)
