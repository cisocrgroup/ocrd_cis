#!/bin/bash
set -e
source $(dirname $0)/test_lib.bash

ocrd_cis_init_ws blumenbach_anatomie_1805.ocrd.zip
# test if there are 3 gt files
pushd "$tmpws"
found_files=0
for file in $(ocrd workspace find -G $OCRD_CIS_FILEGRP); do
	[[ -f "$file" ]] || fail "cannot find ground truth file: $file"
	found_files=$((found_files + 1))
done
(( $found_files == 3 )) || fail "invalid number of files: $found_files"

# download ocr model
ocrd resmgr download ocrd-cis-ocropy-recognize fraktur.pyrnn.gz

# run ocr
ocrd-cis-ocropy-binarize -l DEBUG -I $OCRD_CIS_FILEGRP -O OCR-D-CIS-IMG-BIN
ocrd-cis-ocropy-recognize -l DEBUG -I OCR-D-CIS-IMG-BIN -O OCR-D-CIS-OCR \
	-P textequiv_level word -P model fraktur.pyrnn.gz

popd
