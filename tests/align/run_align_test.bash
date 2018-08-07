#!/bin/bash

set -e

function setup_workspace() {
		get_page_xml_files $TMD_DIR/downloads
		id=1
		JAR=$JAR envsubst < "tests/align/config.json" > $TMP_DIR/config.json
		pushd $TMP_DIR
		ocrd workspace init $TMP_DIR
		for f in $PAGE_XML_FILES; do
				id=$((id+1))
				sid=$(printf '%x' $id)
				echo ocrd workspace add $f -G gt -i gt_$sid -m $PAGE_XML_MIME_TYPE
				ocrd workspace add $f -G gt -i gt_$sid -m $PAGE_XML_MIME_TYPE

				id=$((id+1))
				sid=$(printf '%x' $id)
				sed -e 's/ſ/fl/g' -e 's/ey/ej/g' $f > $f.ocr1
				echo ocrd workspace add $f -G gt -i gt_$sid -m $PAGE_XML_MIME_TYPE
				ocrd workspace add $f.ocr1 -G ocr1 -i ocr1_$sid -m $PAGE_XML_MIME_TYPE

				id=$((id+1))
				sid=$(printf '%x' $id)
				sed -e 's/ſ/j/g' -e 's/ee/ӓ/g' $f > $f.ocr2
				echo ocrd workspace add $f -G gt -i gt_$sid -m $PAGE_XML_MIME_TYPE
				ocrd workspace add $f.ocr2 -G ocr2 -i ocr2_$sid -m $PAGE_XML_MIME_TYPE
		done
		popd
}

source tests/tests.bash
setup_ocrd_test_environment loeber_heuschrecken_1693.zip
setup_workspace

# align 3 file groups
ocrd-cis-align --mets $TMP_DIR/mets.xml \
							 --input-file-grp 'ocr1,ocr2,gt' \
							 --output-file-grp 'ocr1+ocr2+gt' \
							 --parameter file://$TMP_DIR/config.json
assert_file_group_exists 'ocr1+ocr2+gt'
