#!/bin/bash

set -e

function setup_workspace() {
		get_page_xml_files $TMD_DIR/downloads
		id=1
		echo "{\"cisOcrdJar\":\"$TMP_DIR/downloads/ocrd-0.1.jar\"}" > $TMP_DIR/config.json
		pushd $TMP_DIR
		ocrd workspace init $TMP_DIR
		for f in $PAGE_XML_FILES; do
				echo file $f
				id=$((id+1))
				sid=$(printf '%x' $id)
				ocrd workspace add $f -G gt -i gt_$sid -m $PAGE_XML_MIME_TYPE

				id=$((id+1))
				sid=$(printf '%x' $id)
				sed -e 's/ſ/fl/g' -e 's/ey/ej/g' $f > $f.ocr1
				ocrd workspace add $f.ocr1 -G ocr1 -i ocr1_$sid -m $PAGE_XML_MIME_TYPE

				id=$((id+1))
				sid=$(printf '%x' $id)
				sed -e 's/ſ/j/g' -e 's/ee/ӓ/g' $f > $f.ocr2
				ocrd workspace add $f.ocr2 -G ocr2 -i ocr2_$sid -m $PAGE_XML_MIME_TYPE
		done
		popd
}

source tests/tests.bash
setup_ocrd_test_environment loeber_heuschrecken_1693.zip
activate_env
setup_workspace

# align 3 file groups
ocrd-cis-align --mets $TMP_DIR/mets.xml \
							 --input-file-grp 'ocr1,ocr2,gt' \
							 --output-file-grp 'ocr1+ocr2+gt' \
							 --parameter file://$TMP_DIR/config.json
pushd $TMP_DIR
ocrd workspace list-group | grep 'ocr1+ocr2+gt' && true || exit 1
popd
