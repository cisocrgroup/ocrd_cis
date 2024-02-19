#!/bin/bash
set -e
source $(dirname $0)/test_lib.bash

ocrd_cis_init_ws "blumenbach_anatomie_1805.ocrd.zip"

# test if there are 3 gt files
pushd "$tmpws"
found_files=0
for file in $(ocrd workspace find -G $OCRD_CIS_FILEGRP); do
	[[ -f "$file" ]] || fail "cannot find ground truth file: $file"
	found_files=$((found_files + 1))
done
(( found_files == 3 )) || fail "invalid number of files: $found_files"

ocrd-cis-ocropy-binarize -l DEBUG -I $OCRD_CIS_FILEGRP -O OCR-D-CIS-IMG-BIN
ocrd-cis-ocropy-clip -l DEBUG -I OCR-D-CIS-IMG-BIN -O OCR-D-CIS-IMG-CLIP
ocrd-cis-ocropy-denoise -l DEBUG -I OCR-D-CIS-IMG-CLIP -O OCR-D-CIS-IMG-DEN
ocrd-cis-ocropy-deskew -l DEBUG -I OCR-D-CIS-IMG-DEN  -O OCR-D-CIS-IMG-DES
ocrd-cis-ocropy-dewarp -l DEBUG -I OCR-D-CIS-IMG-DES -O OCR-D-CIS-IMG-DEW
ocrd-cis-ocropy-segment -l DEBUG -I OCR-D-CIS-IMG-DEW -O OCR-D-CIS-IMG-SEG
popd
