#!/bin/bash
source ocrd-cis-lib.sh
source $(dirname $0)/test_lib.bash

ocrd_cis_init_ws blumenbach_anatomie_1805.ocrd.zip
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

# download ocr models
wget -P "$tmpdir/download" "http://cis.lmu.de/~finkf/fraktur1-00085000.pyrnn.gz"
wget -P "$tmpdir/download" "http://cis.lmu.de/~finkf/fraktur2-00062000.pyrnn.gz"
# run ocr
ocrd-cis-ocropy-recognize --log-level DEBUG \
						  --input-file-grp "OCR-D-GT-SEG-LINE" \
						  --output-file-grp OCR-D-CIS-OCR-1 \
						  --mets "$tmpws/mets.xml" \
						  --parameter <(cat <<EOF
{
	"textequiv_level": "word",
	"model": "$tmpdir/download/fraktur1-00085000.pyrnn.gz"
}
EOF
)
ocrd-cis-ocropy-recognize --log-level DEBUG \
						  --input-file-grp "OCR-D-GT-SEG-LINE" \
						  --output-file-grp OCR-D-CIS-OCR-2 \
						  --mets "$tmpws/mets.xml" \
						  --parameter <(cat <<EOF
{
	"textequiv_level": "word",
	"model": "$tmpdir/download/fraktur2-00062000.pyrnn.gz"
}
EOF
)
ocrd-cis-align --log-level DEBUG \
			   -I OCR-D-CIS-OCR-1,OCR-D-CIS-OCR-2,OCR-D-GT-SEG-LINE \
			   -O OCR-D-CIS-ALIGN \
			   -m $tmpws/mets.xml

pushd $tmpws
found_files=0
for file in $(ocrd workspace find -G OCR-D-CIS-ALIGN); do
	if [[ ! -f "$file" ]]; then
		echo "cannot find aligned file group workspace"
		exit 1
	fi
	found_files=$((found_files + 1))
done
if [[ $found_files != 3 ]]; then
	echo "invalid number of files: $found_files"
	exit 1
fi
popd
