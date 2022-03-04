[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/cisocrgroup/ocrd_cis.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/cisocrgroup/ocrd_cis/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/cisocrgroup/ocrd_cis.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/cisocrgroup/ocrd_cis/alerts/)
[![image](https://img.shields.io/pypi/v/ocrd_cis.svg)](https://pypi.org/project/ocrd_cis/)

Content:
   * [ocrd_cis](#ocrd_cis)
      * [Introduction](#introduction)
      * [Installation](#installation)
      * [Profiler](#profiler)
      * [Usage](#usage)
         * [ocrd-cis-postcorrect](#ocrd-cis-postcorrect)
         * [ocrd-cis-align](#ocrd-cis-align)
         * [ocrd-cis-data](#ocrd-cis-data)
         * [Training](#training)
         * [ocrd-cis-ocropy-train](#ocrd-cis-ocropy-train)
         * [ocrd-cis-ocropy-clip](#ocrd-cis-ocropy-clip)
         * [ocrd-cis-ocropy-resegment](#ocrd-cis-ocropy-resegment)
         * [ocrd-cis-ocropy-segment](#ocrd-cis-ocropy-segment)
         * [ocrd-cis-ocropy-deskew](#ocrd-cis-ocropy-deskew)
         * [ocrd-cis-ocropy-denoise](#ocrd-cis-ocropy-denoise)
         * [ocrd-cis-ocropy-binarize](#ocrd-cis-ocropy-binarize)
         * [ocrd-cis-ocropy-dewarp](#ocrd-cis-ocropy-dewarp)
         * [ocrd-cis-ocropy-recognize](#ocrd-cis-ocropy-recognize)
         * [Tesserocr](#tesserocr)
      * [Workflow configuration](#workflow-configuration)
      * [Testing](#testing)
   * [Miscellaneous](#miscellaneous)
      * [OCR-D workspace](#ocr-d-workspace)
      * [OCR-D links](#ocr-d-links)

# ocrd_cis

[CIS](http://www.cis.lmu.de) [OCR-D](http://ocr-d.de) command line
tools for the automatic post-correction of OCR-results.

## Introduction
`ocrd_cis` contains different tools for the automatic post-correction
of OCR results.  It contains tools for the training, evaluation and
execution of the post-correction.  Most of the tools are following the
[OCR-D CLI conventions](https://ocr-d.de/en/spec/cli).

Additionally, there is a helper tool to align multiple OCR results,
as well as an improved version of [Ocropy](https://github.com/tmbarchive/ocropy)
that works with Python 3 and is also wrapped for [OCR-D](https://ocr-d.de/en/spec/).

## Installation
There are 2 ways to install the `ocrd_cis` tools:
 * normal packaging:
  ```sh
  make install # or equally: pip install -U pip .
  ```
  (Installs `ocrd_cis` including its Python dependencies
   from the current directory to the Python package directory.)
 * editable mode:
  ```sh
  make install-devel # or equally: pip install -e -U pip .
  ```
  (Installs `ocrd_cis` including its Python dependencies
   from the current directory.)
 
It is possible (and recommended) to install `ocrd_cis` in a custom user directory
(instead of system-wide) by using `virtualenv` (or `venv`):
```sh
 # create venv:
 python3 -m venv venv-dir # where "venv-dir" could be any path name
 # enter venv in current shell:
 source venv-dir/bin/activate
 # install ocrd_cis:
 make install # or any other way (see above)
 # use ocrd_cis:
 ocrd-cis-ocropy-binarize ...
 # finally, leave venv:
 deactivate
```

## Profiler
The post-correction is dependent on the language
[profiler](https://github.com/cisocrgroup/Profiler) and its language
configurations to generate corrections for suspicious words. In order
to use the post-correction, a profiler and according language
configurations have to be present on the system. You can refer to our
[manuals](https://github.com/cisocrgroup/Resources/tree/master/manuals)
and our [lexical
resources](https://github.com/cisocrgroup/Resources/tree/master/lexica)
for more information.

If you use docker you can use the preinstalled profiler from within
the docker-container.  The profiler is installed to `/apps/profiler`
and the language configurations lie in `/etc/profiler/languages` in
the container image.

## Usage
Most tools follow the [OCR-D specifications](https://ocr-d.de/en/spec),
(which makes them [OCR-D _processors_](https://ocr-d.de/en/spec/cli),)
i.e. they accept the command-line options `--input-file-grp`, `--output-file-grp`,
`--page-id`, `--parameter`, `--mets`, `--log-level` (each with an argument).
Invoke with `--help` to get self-documentation. 

Some of the processors (most notably the alignment tool) expect a comma-seperated list
of multiple input file groups, or multiple output file groups.

The [ocrd-tool.json](ocrd_cis/ocrd-tool.json) contains a formal
description of all the processors along with the parameter config file
accepted by their `--parameter` argument.

### ocrd-cis-postcorrect
This processor runs the post correction using a pre-trained model.  If
additional support OCRs should be used, models for these OCR steps are
required and must be executed and aligned beforehand (see [the test
script](tests/run_postcorrection_test.bash) for an example).

There is a basic model trained on the OCR-D ground truth.  It gets
installed allongside this module.  You can get the model's install
path using `ocrd-cis-data -model` (see below for a description of
`ocrd-cis-data`).  To use this model (or any other model) the `model`
parameter in the configuration file must be set to the path of the
model to use.  Be aware that the models are trained with a specific
maximal number of OCR's (usally 2) and that is not possible to use
more OCR's than the number used for training (it is possible to use
less, though).

Arguments:
 * `--parameter` path to configuration file
 * `--input-file-grp` name of the master-OCR file group
 * `--output-file-grp` name of the post-correction file group
 * `--log-level` set log level
 * `--mets` path to METS file in workspace

As mentioned above in order to use the postcorrection with input from
multiple OCR's, some preprocessing steps are needed: firstly the
additional OCR recognition has to be done and secondly the multiple
OCR's have to be aligned (you can also take a look to the function
`ocrd_cis_align` in the [tests](tests/test_lib.bash)).  Assuming an
original recognition as file group `OCR1` on the segmented document of
file group `SEG`, the folloing commands can be used:

```sh
ocrd-ocropus-recognize -I SEG -O OCR2 ... # additional OCR
ocrd-cis-align -I OCR1,OCR2 -O ALGN ... # align OCR1 and OCR2
ocrd-cis-postcorrect -I ALGN -O PC ... # post correction
```

### ocrd-cis-align
Aligns tokens of multiple input file groups to one output file group.
This processor is used to align the master OCR with any additional support
OCRs.  It accepts a comma-separated list of input file groups, which
it aligns in order.

Arguments:
 * `--parameter` path to configuration file
 * `--input-file-grp` comma seperated list of the input file groups;
   first input file group is the master OCR; if there is a ground
   truth (for evaluation) it must be the last file group in the list
 * `--output-file-grp` name of the file group for the aligned result
 * `--log-level` set log level
 * `--mets` path to METS file in workspace

### ocrd-cis-data
Helper tool to get the path of the installed data files. Usage:
`ocrd-cis-data [-h|-jar|-3gs|-model|-config]` to get the path of the
jar library, the pre-trained post correction model, the path to the
default 3-grams language model file or the default training
configuration file.  This tool does not follow the OCR-D conventions.

### Training
There is no dedicated training script provided. Models are trained
using the java implementation directly (check out the [training test
script](tests/run_training_test.bash) for an example).  Training a
model requires a workspace containing one or more file groups
consisting of aligned OCR and ground-truth documents (the last file
group has to be the ground truth).

Arguments:
 * `--parameter` path to configuration file
 * `--input-file-grp` name of the input file group to profile
 * `--output-file-grp` name of the output file group where the profile
   is stored
 * `--log-level` set log level
 * `--mets` path to METS file in the workspace

### ocrd-cis-ocropy-train
The [ocropy-train](ocrd_cis/ocropy/train.py) tool can be used to train LSTM models.
It takes ground truth from the workspace and saves (image+text) snippets from the corresponding pages.
Then a model is trained on all snippets for 1 million (or the given number of) randomized iterations from the parameter file.

```sh
java -jar $(ocrd-cis-data -jar) \
	 -c train \
	 --input-file-grp OCR1,OCR2,GT \
     --log-level DEBUG \
	 -m mets.xml \
	 --parameter $(ocrd-cis-data -config)
```

### ocrd-cis-ocropy-clip
The [clip](ocrd_cis/ocropy/clip.py) processor can be used to remove intrusions of neighbouring segments in regions / lines of a page.
It runs a connected component analysis on every text region / line of every PAGE in the input file group, as well as its overlapping neighbours, and for each binary object of conflict, determines whether it belongs to the neighbour, and can therefore be clipped to the background. It references the resulting segment image files in the output PAGE (via `AlternativeImage`).
(Use this to suppress separators and neighbouring text.)
```sh
ocrd-cis-ocropy-clip \
  -I OCR-D-SEG-REGION \
  -O OCR-D-SEG-REGION-CLIP \
  -p '{"level-of-operation": "region"}'
```

Available parameters are:
```sh
   "level-of-operation" [string - "region"]
    PAGE XML hierarchy level granularity to annotate images for
    Possible values: ["region", "line"]
   "dpi" [number - -1]
    pixel density in dots per inch (overrides any meta-data in the
    images); disabled when negative
   "min_fraction" [number - 0.7]
    share of foreground pixels that must be retained by the largest label
```

### ocrd-cis-ocropy-resegment
The [resegment](ocrd_cis/ocropy/resegment.py) processor can be used to remove overlap between neighbouring lines of a page.
It runs a line segmentation on every text region of every PAGE in the input file group, and for each line already annotated, determines the label of largest extent within the original coordinates (polygon outline) in that line, and annotates the resulting coordinates in the output PAGE.
(Use this to polygonalise text lines that are poorly segmented, e.g. via bounding boxes.)
```sh
ocrd-cis-ocropy-resegment \
  -I OCR-D-SEG-LINE \
  -O OCR-D-SEG-LINE-RES \
  -p '{"extend_margins": 3}'
```

Available parameters are:
```sh
   "level-of-operation" [string - "page"]
    PAGE XML hierarchy level to segment textlines in ('region' abides by
    existing text region boundaries, 'page' optimises lines in the whole
    page once
    Possible values: ["page", "region"]
   "method" [string - "lineest"]
    source for new line polygon candidates ('lineest' for line
    estimation, i.e. how Ocropy would have segmented text lines;
    'baseline' tries to re-polygonize from the baseline annotation;
    'ccomps' avoids crossing connected components by majority rule)
    Possible values: ["lineest", "baseline", "ccomps"]
   "dpi" [number - 0]
    pixel density in dots per inch (overrides any meta-data in the
    images); disabled when zero or negative
   "min_fraction" [number - 0.75]
    share of foreground pixels that must be retained by the output
    polygons
   "extend_margins" [number - 3]
    number of pixels to extend the input polygons in all directions
```

### ocrd-cis-ocropy-segment
The [segment](ocrd_cis/ocropy/segment.py) processor can be used to segment (pages or) regions of a page into (regions and) lines.
It runs a line segmentation on every (page or) text region of every PAGE in the input file group, and adds (text regions containing) `TextLine` elements with the resulting polygon outlines to the annotation of the output PAGE.
(Does _not_ detect tables.)
```sh
ocrd-cis-ocropy-segment \
  -I OCR-D-SEG-BLOCK \
  -O OCR-D-SEG-LINE \
  -p '{"level-of-operation": "page", "gap_height": 0.015}'
```

Available parameters are:
```sh
   "dpi" [number - -1]
    pixel density in dots per inch (overrides any meta-data in the
    images); disabled when negative; when disabled and no meta-data is
    found, 300 is assumed
   "level-of-operation" [string - "region"]
    PAGE XML hierarchy level to read images from and add elements to
    Possible values: ["page", "table", "region"]
   "maxcolseps" [number - 20]
    (when operating on the page/table level) maximum number of
    white/background column separators to detect, counted piece-wise
   "maxseps" [number - 20]
    (when operating on the page/table level) number of black/foreground
    column separators to detect (and suppress), counted piece-wise
   "maximages" [number - 10]
    (when operating on the page level) maximum number of black/foreground
    very large components to detect (and suppress), counted piece-wise
   "csminheight" [number - 4]
    (when operating on the page/table level) minimum height of
    white/background or black/foreground column separators in multiples
    of scale/capheight, counted piece-wise
   "hlminwidth" [number - 10]
    (when operating on the page/table level) minimum width of
    black/foreground horizontal separators in multiples of
    scale/capheight, counted piece-wise
   "gap_height" [number - 0.01]
    (when operating on the page/table level) largest minimum pixel
    average in the horizontal or vertical profiles (across the binarized
    image) to still be regarded as a gap during recursive X-Y cut from
    lines to regions; needs to be larger when more foreground noise is
    present, reduce to avoid mistaking text for noise
   "gap_width" [number - 1.5]
    (when operating on the page/table level) smallest width in multiples
    of scale/capheight of a valley in the horizontal or vertical
    profiles (across the binarized image) to still be regarded as a gap
    during recursive X-Y cut from lines to regions; needs to be smaller
    when more foreground noise is present, increase to avoid mistaking
    inter-line as paragraph gaps and inter-word as inter-column gaps
   "overwrite_order" [boolean - true]
    (when operating on the page/table level) remove any references for
    existing TextRegion elements within the top (page/table) reading
    order; otherwise append
   "overwrite_separators" [boolean - true]
    (when operating on the page/table level) remove any existing
    SeparatorRegion elements; otherwise append
   "overwrite_regions" [boolean - true]
    (when operating on the page/table level) remove any existing
    TextRegion elements; otherwise append
   "overwrite_lines" [boolean - true]
    (when operating on the region level) remove any existing TextLine
    elements; otherwise append
   "spread" [number - 2.4]
    distance in points (pt) from the foreground to project text line (or
    text region) labels into the background for polygonal contours; if
    zero, project half a scale/capheight
```

### ocrd-cis-ocropy-deskew
The [deskew](ocrd_cis/ocropy/deskew.py) processor can be used to deskew pages / regions of a page.
It runs a projection profile-based skew estimation on every segment of every PAGE in the input file group and annotates the orientation angle in the output PAGE.
(Does _not_ include orientation detection.)
```sh
ocrd-cis-ocropy-deskew \
  -I OCR-D-SEG-LINE \
  -O OCR-D-SEG-LINE-DES \
  -p '{"level-of-operation": "page", "maxskew": 10}'
```

Available parameters are:
```sh
   "maxskew" [number - 5.0]
    modulus of maximum skewing angle to detect (larger will be slower, 0
    will deactivate deskewing)
   "level-of-operation" [string - "region"]
    PAGE XML hierarchy level granularity to annotate images for
    Possible values: ["page", "region"]
```

### ocrd-cis-ocropy-denoise
The [denoise](ocrd_cis/ocropy/denoise.py) processor can be used to despeckle pages / regions / lines of a page.
It runs a connected component analysis and removes small components (black or white) on every segment of every PAGE in the input file group and references the resulting segment image files in the output PAGE (as `AlternativeImage`).
```sh
ocrd-cis-ocropy-denoise \
  -I OCR-D-SEG-LINE-DES \
  -O OCR-D-SEG-LINE-DEN \
  -p '{"noise_maxsize": 2}'
```

Available parameters are:
```sh
   "noise_maxsize" [number - 3.0]
    maximum size in points (pt) for connected components to regard as
    noise (0 will deactivate denoising)
   "dpi" [number - -1]
    pixel density in dots per inch (overrides any meta-data in the
    images); disabled when negative
   "level-of-operation" [string - "page"]
    PAGE XML hierarchy level granularity to annotate images for
    Possible values: ["page", "region", "line"]
```

### ocrd-cis-ocropy-binarize
The [binarize](ocrd_cis/ocropy/binarize.py) processor can be used to binarize (and optionally denoise and deskew) pages / regions / lines of a page.
It runs the "nlbin" adaptive whitelevel thresholding on every segment of every PAGE in the input file group and references the resulting segment image files in the output PAGE (as `AlternativeImage`). (If a deskewing angle has already been annotated in a region, the tool respects that and rotates accordingly.) Images can also be produced grayscale-normalized.
```sh
ocrd-cis-ocropy-binarize \
  -I OCR-D-SEG-LINE-DES \
  -O OCR-D-SEG-LINE-BIN \
  -p '{"level-of-operation": "page", "threshold": 0.7}'
```

Available parameters are:
```sh
   "method" [string - "ocropy"]
    binarization method to use (only 'ocropy' will include deskewing and
    denoising)
    Possible values: ["none", "global", "otsu", "gauss-otsu", "ocropy"]
   "threshold" [number - 0.5]
    for the 'ocropy' and ' global' method, black/white threshold to apply
    on the whitelevel normalized image (the larger the more/heavier
    foreground)
   "grayscale" [boolean - false]
    for the 'ocropy' method, produce grayscale-normalized instead of
    thresholded image
   "maxskew" [number - 0.0]
    modulus of maximum skewing angle (in degrees) to detect (larger will
    be slower, 0 will deactivate deskewing)
   "noise_maxsize" [number - 0]
    maximum pixel number for connected components to regard as noise (0
    will deactivate denoising)
   "level-of-operation" [string - "page"]
    PAGE XML hierarchy level granularity to annotate images for
    Possible values: ["page", "region", "line"]
```

### ocrd-cis-ocropy-dewarp
The [dewarp](ocrd_cis/ocropy/dewarp.py) processor can be used to vertically dewarp text lines of a page.
It runs the baseline estimation and center normalizer algorithm on every line in every text region of every PAGE in the input file group and references the resulting line image files in the output PAGE (as `AlternativeImage`).
```sh
ocrd-cis-ocropy-dewarp \
  -I OCR-D-SEG-LINE-BIN \
  -O OCR-D-SEG-LINE-DEW \
  -p '{"range": 5}'
```

Available parameters are:
```sh
   "dpi" [number - -1]
    pixel density in dots per inch (overrides any meta-data in the
    images); disabled when negative
   "range" [number - 4.0]
    maximum vertical disposition or maximum margin (will be multiplied by
    mean centerline deltas to yield pixels)
   "max_neighbour" [number - 0.05]
    maximum rate of foreground pixels intruding from neighbouring lines
    (line will not be processed above that)
```

### ocrd-cis-ocropy-recognize
The [recognize](ocrd_cis/ocropy/recognize.py) processor can be used to recognize the lines / words / glyphs of a page.
It runs LSTM optical character recognition on every line in every text region of every PAGE in the input file group and adds the resulting text annotation in the output PAGE.
```sh
ocrd-cis-ocropy-recognize \
  -I OCR-D-SEG-LINE-DEW \
  -O OCR-D-OCR-OCRO \
  -p '{"textequiv_level": "word", "model": "fraktur-jze.pyrnn"}'
```

Available parameters are:
```sh
   "textequiv_level" [string - "line"]
    PAGE XML hierarchy level granularity to add the TextEquiv results to
    Possible values: ["line", "word", "glyph"]
   "model" [string]
    ocropy model to apply (e.g. fraktur.pyrnn)
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
https://github.com/tesseract-ocr/tesseract/wiki/Data-Files or use your
own models and place them into: /usr/share/tesseract-ocr/4.00/tessdata

## Workflow configuration

A decent pipeline might look like this:

1. image normalization/optimization
1. page-level binarization
1. page-level cropping
1. (page-level binarization)
1. (page-level despeckling)
1. page-level deskewing
1. (page-level dewarping)
1. region segmentation, possibly subdivided into
   1. text/non-text separation
   1. text region segmentation (and classification)
   1. reading order detection
   1. non-text region classification
1. region-level clipping
1. (region-level deskewing)
1. line segmentation
1. (line-level clipping or resegmentation)
1. line-level dewarping
1. line-level recognition
1. (line-level alignment and post-correction)

If GT is used, then cropping/segmentation steps can be omitted.

If a segmentation is used which does not produce overlapping segments, then clipping/resegmentation can be omitted.

## Testing
To run a few basic tests type `make test` (`ocrd_cis` has to be
installed in order to run any tests).

# Miscellaneous
## OCR-D workspace

* Create a new (empty) workspace: `ocrd workspace -d workspace-dir init`
* cd into `workspace-dir`
* Add new file to workspace: `ocrd workspace add file -G group -i id
  -m mimetype -g pageId`

## OCR-D links

- [OCR-D](https://ocr-d.github.io)
- [Github](https://github.com/OCR-D)
- [Project-page](http://www.ocr-d.de/)
- [Ground-truth](https://ocr-d-repo.scc.kit.edu/api/v1/metastore/bagit/search)
