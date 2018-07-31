#!/bin/bash

set -e

function setup_workspace() {
		get_page_xml_files $TMD_DIR/downloads
		id=1
		local pexe="$HOME/devel/work/Profiler/build/bin/profiler"
		local pback="$HOME/data/profiler-backend"
		JAR=$JAR pexe=$pexe pback=$pback plang="german" envsubst \
			 < "tests/profile/config.json" \
			 > $TMP_DIR/config.json
		pushd $TMP_DIR
		ocrd workspace init $TMP_DIR
		for f in $PAGE_XML_FILES; do
				id=$((id+1))
				sid=$(printf '%x' $id)
				echo ocrd workspace add $f -G gt -i gt_$sid -m $PAGE_XML_MIME_TYPE
				ocrd workspace add $f -G gt -i gt_$sid -m $PAGE_XML_MIME_TYPE
		done
		popd
}

source tests/tests.bash
setup_ocrd_test_environment loeber_heuschrecken_1693.zip
activate_env
setup_workspace

# profile 3 files
ocrd-cis-profile --mets $TMP_DIR/mets.xml \
							 --input-file-grp 'gt' \
							 --output-file-grp 'profiled-gt' \
							 --parameter file://$TMP_DIR/config.json
# pushd $TMP_DIR
# ocrd workspace list-group | grep 'profiled-gt' && true || exit 1
# popd
