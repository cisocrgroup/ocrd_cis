#!/bin/bash

set -e

PAGE_XML_MIME_TYPE="application/vnd.prima.page+xml"
CACHE_DIR="/tmp/cis-ocrd-py-cache"
TMP_DIR=$(mktemp -d -t cis-ocrd-align-XXXXXXXXX)

function rmtd() {
		echo removing $TMP_DIR
		rm -rf $TMP_DIR
}
trap rmtd EXIT

function wget_cached() {
		local url=$1
		local filename=$2
		local destdir=$TMP_DIR/downloads

		if test ! -f $CACHE_DIR/$filename; then
				mkdir -p $CACHE_DIR
				echo "downloading $url/$filename"
				wget -P $CACHE_DIR $url/$filename
		fi
		mkdir -p $destdir
		ln $CACHE_DIR/$filename $destdir/$filename ||\
				cp $CACHE_DIR/$filename $destdir/$filename
}

function download_ocrd_gt_zip() {
		local url="http://www.ocr-d.de/sites/all/GTDaten"
		local filename=$1
		wget_cached $url $filename
}

function unzip_ocrd_gt() {
		local zip="$TMP_DIR/downloads/$1"
		echo unziping $zip
		unzip -d $TMP_DIR/downloads ${zip/.zip/} >/dev/null
}

function download_and_unzip_ocrd_gt() {
		download_ocrd_gt_zip $1
		unzip_ocrd_gt $1
}

function get_page_xml_files() {
		page_xml_files=""
		for d in $(find $TMP_DIR/downloads -type d -name page); do
				for f in $(find $d -type f | sort); do
						page_xml_files+=" $f"
				done
		done
}

function activate_env() {
		local envdir=$CACHE_DIR/env
		if test ! -d $envdir; then
				virtualenv -p python3.6 $envdir
				source $envdir/bin/activate
				pip install -r requirements.txt
				pip install .
		else
				source $envdir/bin/activate
		fi
}

function download_ocrd_jar() {
		local url='http://www.cis.lmu.de/~finkf'
		wget_cached $url "ocrd-0.1.jar"
}

function setup_ocrd_test_environment() {
		download_and_unzip_ocrd_gt $1
		download_ocrd_jar
		activate_env
}

function setup_workspace() {
		get_page_xml_files
		id=1
		echo "{\"cisOcrdJar\":\"$TMP_DIR/downloads/ocrd-0.1.jar\"}" > $TMP_DIR/config.json
		pushd $TMP_DIR
		ocrd workspace init $TMP_DIR
		for f in $page_xml_files; do
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
