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
(( found_files == 3 )) || fail "invalid number of files: $found_files"
popd

ocrd_cis_align

pushd $tmpws
found_files=0
for file in $(ocrd workspace find -G OCR-D-CIS-ALIGN); do
	[[ -f "$file" ]] || fail "cannot find aligned file group workspace"
	found_files=$((found_files + 1))
done
(( found_files == 3 )) || fail "invalid number of files: $found_files"
popd
