#!/bin/bash

set -e
# global default log-level
LOG_LEVEL=DEBUG

ocrd-cis-log() {
	echo $(date +%R:%S.%N | sed -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]\).*/\1/') $LOG_LEVEL $* >&2
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

# Download the ocrd.jar if it does not exist.
ocrd-cis-download-jar() {
	if [[ -f "$1" ]]; then
		return 0
	fi
	local jar=http://www.cis.lmu.de/~finkf/ocrd.jar
	local dir=$(/usr/bin/dirname $1)
	pushd $dir
	wget -N $jar || true
	popd
}

# Get the mimetype of a given path.  The mimetype is determined using
# the file's extension.
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
# $OCRFILEGRPS to a space-separated list of the ocr output file
# groups.
ocrd-cis-run-ocr() {
	local config=$1
	local mets=$2
	local ifg=$3
	local ofg=$4
	OCRFILEGRPS=""

	for i in $(seq 0 $(cat "$config" | jq ".ocr | length-1")); do
		local type=$(cat "$config" | jq --raw-output ".ocr[$i].type")
		local path=$(cat "$config" | jq --raw-output ".ocr[$i].path")
		local utype=$(echo $type | tr '[:lower:]' '[:upper:]')
		local xofg=${ofg/XXX/$utype-$((i+1))}
		OCRFILEGRPS="$OCRFILEGRPS $xofg"
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
				ocrd-cis-log ocrd-tesserocr-recognize \
					--input-file-grp $ifg \
					--output-file-grp $xofg \
					--mets "$mets" \
					--parameter $path \
					--log-level $LOG_LEVEL
				ocrd-tesserocr-recognize \
					--input-file-grp $ifg \
					--output-file-grp $xofg \
					--mets "$mets" \
					--parameter $path \
					--log-level $LOG_LEVEL
				;;
			*)
				echo "invalid ocr type: $utype"
				exit 1
				;;
		esac
	done
}

# Search for the associated image file for the given xml file in the
# given directory.  The given xml file must end with .xml. Usage:
# `ocrd-cis-find-image-for-xml dir xy.xml`
ocrd-cis-find-image-for-xml() {
	local dir=$1
	local xml=$2

	for pre in .bin .dew ""; do # prefer binary before descewed before normal images
		for ext in .jpg .jpeg .JPG .JPEG .png .tiff .tif; do
			# strict search based on the xml file's name
			# try also using the xml file's number, e.g xyz_123.xml -> 123.bin.png
			local name=$(basename "$xml")
			local name=${name/.xml/$pre$ext}
			local numname=$(echo $name | sed -e 's/.*[-_]\([0-9]*\.\)/\1/')
			local file=$(find "$dir" -type f -name "$name" -o -type f -name "$numname")
			# echo find "$dir" -type f -name "$name" -o -type f -name "$numname"
			# echo file $file
			if [[ ! -z "$file" ]]; then
				ocrd-cis-log "[$xml]" found $file
				echo $file
				return 0
			fi
		done
	done
	return 1
}

# Add a pagexml and image file pair to a workspace.  The according
# imageFilename attribute of the page xml file is set accordingly.
# The basename of the given files are used as file ids.  Usage:
# `ocrd-cis-add-xml-image-pair mets xml xmlfg img imgfg`.
# * mets:   path to the workspace's mets file
# * xml:    path to the page xml file
# * xmlfg:  file group of the xml file
# * img:    path to the imaage file
# * imgfg:  file group of the image file
ocrd-cis-add-xml-image-pair() {
	local mets=$1
	local xml=$2
	local xmlfg=$3
	local img=$4
	local imgfg=$5

	local imgmt=$(ocrd-cis-get-mimetype-by-extension "$img")
	local xmlmt=$(ocrd-cis-get-mimetype-by-extension "$xml")
	local workspace=$(dirname "$mets")
	local absxml=$(realpath "$xml")
	local absimg=$(realpath "$img")

	pushd $workspace
	# add image file
	ocrd workspace add \
		 --file-grp "$imgfg" \
		 --mimetype "$imgmt" \
		 --file-id "$(basename "$img")" \
		 --force "$absimg"
	# add xml file
	ocrd workspace add \
		 --file-grp "$xmlfg" \
		 --mimetype "$xmlmt" \
		 --file-id "$(basename "$xml")" \
		 --force "$absxml"
	# fix filepath
	local relxml="OCR-D-$gt-$fg/$(basename $xml)"
	local relimg="OCR-D-IMG-$fg/$(basename $img)"
	sed -i "s#imageFilename=\"\([^\"]*\)\"#imageFilename=\"$relimg\"#" "$relxml"
	popd
}

# Given a directory add image and base xml files, run additional ocrs
# and align them.  Sets ALGINFILEGRP to the alignment file group.
# Usage: `ocrd-cis-run-ocr-and-align config mets dir fg gt`.
# * config	: path to the main config file
# * mets		: path to the mets file
# * dir		: path to the directory
# * fg		: base name of filegroups
# * gt		: gt=GT if xml files are ground truth; anythin else if not
ocrd-cis-run-ocr-and-align() {
	local config=$1
	local mets=$2
	local dir=$3
	local fg=$4
	local gt=$5
	local workspace=$(dirname "$mets")

	for xml in $(find "$dir" -type f -name '*.xml'); do
		if [[ "$xml" == *"alto"* ]]; then # skip alto xml files in gt archives
		   continue
		fi
		local img=$(ocrd-cis-find-image-for-xml "$dir" "$xml")
		ocrd-cis-add-xml-image-pair "$mets" "$xml" "OCR-D-$gt-$fg" "$img" "OCR-D-IMG-$fg"
	done
	OCRFILEGRPS=""
	ocrd-cis-run-ocr "$config" "$mets" "OCR-D-$gt-$fg" "OCR-D-XXX-$fg"
	if [[ $(echo "$gt" | tr '[[:upper:]]' '[[:lower:]]') == "gt" ]]; then
		OCRFILEGRPS="$OCRFILEGRPS OCR-D-$gt-$fg"
	else
		OCRFILEGRPS="OCR-D-$gt-$fg $OCRFILEGRPS"
	fi
	OCRFILEGRPS=$(ocrd-cis-join-by , $OCRFILEGRPS)
	ALIGNFILEGRP="OCR-D-ALIGN-$fg"
	ocrd-cis-log ocrd-cis-align \
		--input-file-grp "$OCRFILEGRPS" \
		--output-file-grp "$ALIGNFILEGRP" \
		--mets "$mets" \
		--parameter $(cat "$config" | jq --raw-output ".alignparampath") \
		--log-level $LOG_LEVEL
	ocrd-cis-align \
		--input-file-grp "$OCRFILEGRPS" \
		--output-file-grp "$ALIGNFILEGRP" \
		--mets "$mets" \
		--parameter $(cat "$config" | jq --raw-output ".alignparampath") \
		--log-level $LOG_LEVEL
}
