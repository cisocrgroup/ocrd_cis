#!/bin/bash
source ocrd-cis-lib.sh
source $(dirname $0)/test_lib.bash

ocrd_cis_init_ws "blumenbach_anatomie_1805.ocrd.zip"

# test if there are 3 gt files
pushd "$tmpws"
found_files=0
for file in $(ocrd workspace find -G OCR-D-GT-SEG-LINE); do
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

ocrd-cis-ocropy-binarize --log-level DEBUG \
						 --input-file-grp OCR-D-GT-SEG-LINE \
						 --output-file-grp OCR-D-CIS-IMG-BIN \
						 --mets "$tmpws/mets.xml"

ocrd-cis-ocropy-clip --log-level DEBUG \
					 --input-file-grp OCR-D-CIS-IMG-BIN \
					 --output-file-grp OCR-D-CIS-IMG-CLIP \
					 --mets "$tmpws/mets.xml"

ocrd-cis-ocropy-denoise --log-level DEBUG \
						--input-file-grp OCR-D-CIS-IMG-CLIP \
						--output-file-grp OCR-D-CIS-IMG-DEN \
						--mets "$tmpws/mets.xml"

ocrd-cis-ocropy-deskew --log-level DEBUG \
					   --input-file-grp OCR-D-CIS-IMG-DEN \
					   --output-file-grp OCR-D-CIS-IMG-DES \
					   --mets "$tmpws/mets.xml"

ocrd-cis-ocropy-dewarp --log-level DEBUG \
					   --input-file-grp OCR-D-CIS-IMG-DES \
					   --output-file-grp OCR-D-CIS-IMG-DEW \
					   --mets "$tmpws/mets.xml"
