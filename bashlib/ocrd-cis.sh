#!/bin/bash

set -e

ocrd-cis-log() {
	echo LOG: $* >&2
}

# utility function to join strings with a given string
function ocrd-cis-join-by { local IFS="$1"; shift; echo "$*"; }

# Parse command line arguments for a given argument and returns its
# value.  Usage: `ocrd-cis-getopt -P --parameter $*`.
ocrd-cis-getopt() {
	short=$1
	shift
	long=$1
	shift
	while [[ $# -gt 0 ]]; do
		case $1 in
			$short|$long) echo $2; return 0;;
			*) shift;;
		esac
	done
	ocrd-cis-log "missing command line argument: $short | $long"
	exit 1
}

# Download the ocrd.jar.
ocrd-cis-download-jar() {
	local jar=http://www.cis.lmu.de/~finkf/ocrd.jar
	local dir=$(/usr/bin/dirname $1)
	pushd $dir
	wget -N $jar || true
	popd
}

# Add OCR page XML and its image to the workspace. Usage:
# `ocrd-cis-add-pagexml-and-image-to-workspace workspace pagexmlfg
# pagexml imagefg image`.
ocrd-cis-add-pagexml-and-image-to-workspace() {
	local workspace=$1
	local pagexmlfg=$2
	local pagexml=$3
	local imagefg=$4
	local image=$5

	pushd "$workspace"
	# add image
	local mime=$(ocrd-cis-get-mimetype-by-extension $image)
	local fileid=$(basename $image)
	local addpath="$imagefg/$fileid"
	ocrd-cis-log ocrd workspace add --file-grp "$imagefg" --file-id "$fileid" --mimetype "$mime" "../$image"
	ocrd workspace add --file-grp "$imagefg" --file-id "$fileid" --mimetype "$mime" "../$image"

	# add page xml
	local mime=$(ocrd-cis-get-mimetype-by-extension $pagexml)
	local fileid=$(basename $pagexml)
	ocrd-cis-log ocrd workspace add --file-grp "$pagexmlfg" --file-id "$fileid" --mimetype "$mime" "../$pagexml"
	ocrd workspace add --file-grp "$pagexmlfg" --file-id "$fileid" --mimetype "$mime" "../$pagexml"

	# fix imageFilepath in page xml
	local absimgpath=$(realpath $addpath)
	sed -i "$pagexmlfg/$fileid" -e "s#imageFilename=\"[^\"]*\"#imageFilename=\"$absimgpath\"#"
	popd
}

ocrd-cis-get-mimetype-by-extension() {
	case $(echo $1 | tr '[:upper:]' '[:lower:]') in
		*.tif | *.tiff) echo "image/tif";;
		*.jpg | *.jpeg) echo "image/jpeg";;
		*.png) echo "image/png";;
		*.xml) echo "application/vnd.prima.page+xml";;
		*) echo "UNKWNON"
	esac
}

# Run multiple OCRs over a file group.  Usage: `ocrd-cis-run-ocr
# configfile mets ifg ofg`.  A XXX in the ofg is replaced with the
# ocr-type and number.  This function sets the global variable
# $ALIGNFILEGRPS to a space-separated list of the ocr output file
# groups.
ocrd-cis-run-ocr() {
	local config=$1
	local mets=$2
	local ifg=$3
	local ofg=$4
	ALIGNFILEGRPS=""

	for i in $(seq 0 $(cat "$config" | jq ".ocr | length-1")); do
		type=$(cat "$config" | jq --raw-output ".ocr[$i].type")
		path=$(cat "$config" | jq --raw-output ".ocr[$i].path")
		utype=$(echo $type | tr '[:lower:]' '[:upper:]')
		xofg=${ofg/XXX/$utype-$((i+1))}
		ALIGNFILEGRPS="$ALIGNFILEGRPS $xofg"
		case $utype in
			"OCROPY")
				ocrd-cis-log ocrd-cis-ocropy-recognize \
					--input-file-grp $ifg \
					--output-file-grp $xofg \
					--mets "$mets" \
					--parameter $path \
					--log-level $LOG_LEVEL
				ocrd-cis-ocropy-recognize \
					--input-file-grp $ifg \
					--output-file-grp $xofg \
					--mets "$mets" \
					--parameter $path \
					--log-level $LOG_LEVEL
				;;
			"TESSERACT")
						echo "not implemented: tesseract $path"
				exit 1
				;;
			*)
				echo "invalid ocr type: $utype"
				exit 1
				;;
		esac
	done
}

# Search for the associated image file for the given xml file in the
# given directory. The given xml file must end with .xml. Usage:
# `ocrd-cis-find-image-for-xml dir xy.xml`
ocrd-cis-find-image-for-xml() {
	local dir=$1
	local xml=$2

	for pre in "" .dew .bin; do
		for ext in .jpg .jpeg .JPG .JPEG .png .tiff; do
			local name=${xml/.xml/$pre$ext}
			local file=$(find $dir -type f -name $name)
			if [[ ! -z $file ]]; then
				ocrd-cis-log "[$xml]" found $file
				echo $file
				return 0
			fi
		done
	done
	return 1
}
