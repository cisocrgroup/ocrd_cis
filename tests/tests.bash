#!/bin/bash

set -e

PAGE_XML_MIME_TYPE="application/vnd.prima.page+xml"
CACHE_DIR="/tmp/cis-ocrd-py-cache"
TMP_DIR=$(mktemp -d -t cis-ocrd-align-XXXXXXXXX)
PAGE_XML_FILES=""
JAR=""

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
		PAGE_XML_FILES=""
		for d in $@; do
				for d in $(find $TMP_DIR/downloads -type d -name page); do
						for f in $(find $d -type f | sort); do
								PAGE_XML_FILES+=" $f"
						done
				done
		done
}

function download_ocrd_jar() {
		local url='http://www.cis.lmu.de/~finkf'
		wget_cached $url "ocrd-0.1.jar"
		JAR="$TMP_DIR/downloads/ocrd-0.1.jar"
}

function setup_ocrd_test_environment() {
		download_and_unzip_ocrd_gt $1
		download_ocrd_jar
}

function assert_file_group_exists() {
		pushd $TMP_DIR
		ocrd workspace list-group | grep "$1" && true || exit 1
		popd
}
