#!/bin/bash
source ocrd-cis-lib.sh
source $(dirname $0)/test_lib.sh

url="http://www.ocr-d.de/sites/all/GTDaten/blumenbach_anatomie_1805.zip"
mkdir -p "$tmpdir/download"
pushd "$tmpws"
ocrd-cis-download-and-add-gt-zip "$url" "$tmpdir/download"
popd

# test if there are 3 gt files
pushd "$tmpws"
found_files=0
for file in $(ocrd workspace find -G "$OCR_D_CIS_GT_FILEGRP"); do
	if [[ ! -f "$file" ]]; then
		echo "cannot find ground truth file: $file"
		exit 1
	fi
	found_files=$((found_files + 1))
done
if [[ $found_files != 3 ]]; then
	echo "invalid number of files: $found_files"
	exit 1
fi
popd

# train ocr
ocrd-cis-ocropy-train --log-level DEBUG \
					  --input-file-grp "$OCR_D_CIS_GT_FILEGRP" \
					  --mets "$tmpws/mets.xml" \
					  --parameter <(echo '{"ntrain": 50}')
