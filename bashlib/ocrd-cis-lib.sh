#!/bin/bash

set -e
# global default log-level
LOG_LEVEL=DEBUG

ocrd-cis-log() {
	echo $(date +%R:%S.%N | sed -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]\).*/\1/') $LOG_LEVEL - $* >&2
}

# Write a OCR-D debug log message to stderr.
ocrd-cis-debug() {
	case $LOG_LEVEL in
		DEBUG) echo $(date +%R:%S.%N | sed -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]\).*/\1/') DEBUG ocrd.cis.bashlib - $* >&2;;
	esac
}

# Write a OCR-D info log message to stderr.
ocrd-cis-info() {
	case $LOG_LEVEL in
		DEBUG|INFO) echo $(date +%R:%S.%N | sed -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]\).*/\1/') INFO ocrd.cis.bashlib - $* >&2;;
	esac
}

# Print error message to stderr and exit.
# Usage `ocrd-cis-fail "error message" [EXIT]`
function ocrd-cis-fail {
	printf '%s\n' "$1" >&2
	exit "${2-1}"
}

# utility function to join strings with a given string
function ocrd-cis-join-by { local IFS="$1"; shift; echo "$*"; }

# Parse command line arguments for a given argument and
# SETS_CIS_OPTARG to the additional provided value.  Usage:
# `ocrd-cis-getopt -P --parameter $*`.
ocrd-cis-getopt() {
	OCRD_CIS_OPTARG=""
	local short=$1
	shift
	local long=$1
	shift
	while [[ $# -gt 0 ]]; do
		case $1 in
			$short|$long) OCRD_CIS_OPTARG=$2; return 0;;
			*) shift;;
		esac
	done
	return 1;
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

# Get the file for a file path.  Sets OCRD_CIS_FILE_ID to
# the file id.  Usage: ocrd-cis-file-id path/to/file.xml
OCRD_CIS_FILE_ID=""
ocrd-cis-get-file-id() {
	local path=$1
	local filename=$(basename "$path")
	local ext="${filename##*.}"
	local fileid="${filename%.*}"
	OCRD_CIS_FILE_ID="${fileid}_${ext}"
	echo $path $OCRD_CIS_FILE_ID
}

# Add a zipped OCR-D ground truth zip to the workspace.  The current
# directory must be a valid workspace with an according mets file.
# Sets OCR_D_CIS_GT_FILEGRP and OCR_D_CIS_IMG_FILEGRP to the according
# filegroups.  Exits if the image file for a page-XML file in the zip
# archive cannot be found.
# Usage: `ocrd-cis-add-gt-zip ZIP TMP_DIR
# * ZIP:     path to the gt-zip file
# * TMP_DIR: existing temporary directory for extracted files
ocrd-cis-add-gt-zip() {
	local zip=$1
	local tmp=$2
	ocrd-cis-log ocrd-cis-add-gt-zip $zip $tmp
	unzip -d "$tmp" "$zip"

	local base=$(echo $(basename $zip) | tr '_ \t' '-')
	base=${base/.zip/}
	local gtfg="OCR-D-GT-$base"
	local imgfg="OCR-D-IMG-$base"
	for xml in $(find "$tmp" -type f -name '*.xml' | grep -i 'page'); do
		local imgname=$(sed -ne 's/.*imageFilename="\([^"]*\)".*/\1/p' "$xml")
		local img=$(find "$tmp" -type f -name "$imgname")
		if [[ ! -f "$img" ]]; then
			echo "cannot find image: $imgname"
			exit 1
		fi
		# add image to workspace
		local imgmimetype=$(ocrd-cis-get-mimetype-by-extension "$img")
		ocrd-cis-get-file-id "$img"
		ocrd workspace add \
			 --file-grp "$imgfg" \
			 --mimetype "$imgmimetype" \
			 --file-id "$OCRD_CIS_FILE_ID" \
			 "$img"
		# get img path in workspace and set imageFilename in page xml accordingly.
		img=$(ocrd workspace find -i "$OCRD_CIS_FILE_ID")
		sed -i -e "s#imageFilename=\"[^\"]*\"#imageFilename=\"$img\"#" "$xml"
		# add page xml file to workspace
		ocrd-cis-get-file-id "$xml"
		ocrd workspace add \
			 --file-grp "$gtfg" \
			 --mimetype "application/vnd.prima.page+xml" \
			 --file-id "$OCRD_CIS_FILE_ID" \
			 "$xml"
	done
	# set global filegroup variables
	OCR_D_CIS_GT_FILEGRP=$gtfg
	OCR_D_CIS_IMG_FILEGRP=$imgfg
}

# Add a zipped OCR-D ground truth zip to the workspace.  The current
# directory must be a valid workspace with an according mets file.
# Usage: `ocrd-cis-add-gt-zip URL TMP_DIR
# * URL:     URL to the gt-zip file
# * TMP_DIR: existing temporary directory for downloaded (and
#   extracted) files
ocrd-cis-download-and-add-gt-zip() {
	local url=$1
	local tmp=$2
	ocrd-cis-log ocrd-cis-download-and-add-gt-zip $url $tmp
	wget -P "$tmp" $url
	local zip=$(find $tmp -type f -name '*.zip')
	echo $zip
	ocrd-cis-add-gt-zip "$zip" "$tmp"
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

# Check if given file-id exists in the given mets file.  Usage:
# `ocrd-cis-file-id-exists mets fileid`.
# * mets:   path to the mets file
# * fileid: the file id
ocrd-cis-file-id-exists() {
	local workspace=$(dirname "$1")
	local fileid=$2
	pushd "$workspace"
	local check=$(ocrd workspace find --file-id "$fileid")
	popd
	if [[ -z "$check" ]]; then return 1; fi
	return 0
}

# Check if given file-grp exists in the given mets file.  Usage:
# `ocrd-cis-file-grp-exists mets filegrp`.
# * mets:    path to the mets file
# * filegrp: the file grp
ocrd-cis-file-grp-exists() {
	local workspace=$(dirname "$1")
	local filegrp=$2
	pushd "$workspace"
	local check=$(ocrd workspace find --file-grp "$filegrp")
	popd
	if [[ -z "$check" ]]; then return 1; fi
	return 0
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
		if ocrd-cis-file-grp-exists "$mets" "$xofg"; then
			ocrd-cis-log skipping ocr for $xofg
			continue
		# else
		# 	ocrd-cis-log $xofg does not exist.
		# 	exit 1
		fi
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
				ocrd-cis-log found image: $file for xml: $xml
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

	if ocrd-cis-file-id-exists "$mets" "$(basename "$img")"; then
		ocrd-cis-log skipping add to workspace for $img and $xml
		return
	fi

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
	local relxml="$xmlfg/$(basename $xml)"
	local relimg="$imgfg/$(basename $img)"
	echo sed -i "s#imageFilename=\"\([^\"]*\)\"#imageFilename=\"$relimg\"#" "$relxml"
	sed -i "s#imageFilename=\"\([^\"]*\)\"#imageFilename=\"$relimg\"#" "$relxml"
	popd
}

# Given a directory add image and base xml files, run additional ocrs
# and align them.  Sets ALGINFILEGRP to the alignment file group.
# Usage: `ocrd-cis-run-ocr-and-align config mets dir fg gt`.
# * config	: path to the main config file
# * mets	: path to the mets file
# * dir		: path to the directory
# * fg		: base name of filegroups
# * gt		: gt=GT if xml files are ground truth; anything else if not
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
	if ocrd-cis-file-grp-exists "$mets" "$ALIGNFILEGRP"; then
		ocrd-cis-log skipping aligning of $ALIGNFILEGRP
		return
	fi
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
	# (Cannot use non unicode chars if installing this)
	# Change long s (\u017f) to normal s if the ground truth
	# does not contain long s.
	# fixlongs=$(cat "$config" | jq --raw-output '.fixLongS')
	# if [[ "$fixlongs" == "true" ]]; then
	# 	pushd "$workspace"
	# 	ocrd-cis-log "fixing long s in file"
	# 	for fg in $(ocrd workspace list-group | grep 'ALIGN'); do
	# 		ocrd-cis-log "fixing long s in filegroup $fg"
	# 		for xml in "$fg"/*; do
	# 			ocrd-cis-log "fixing long s in file $xml"
	# 			sed -i -e 's/\u017f/s/g' "$xml"
	# 		done
	# 	done
	# 	popd
	# fi
}

# Run the training over the `-ALIGN-` filegroups in the workspace
# directory of the given mets.xml file.  Usage: `ocrd-cis-run-training
# config mets`.
# * config: path to the configuration file
# * mets:   path to the mets file
ocrd-cis-run-training() {
	local config=$1
	local mets=$2
	local workspace=$(dirname "$mets")
	local main="de.lmu.cis.ocrd.cli.Main"
	local jar=$(cat "$config" | jq --raw-output '.jar')
	local trainconfig=$(cat "$config" | jq --raw-output '.trainparampath')

	# get -ALIGN- filegroups
	pushd "$workspace"
	local trainfilegrps=""
	for fg in $(ocrd workspace list-group); do
		if [[ $fg == *"-ALIGN-"* ]]; then
			trainfilegrps="$trainfilegrps -I $(basename $fg)"
		fi
	done
	popd
	# run training
	ocrd-cis-log java -Dfile.encoding=UTF-8 -Xmx3g -cp $jar $main --log-level $LOG_LEVEL \
				 -c train --mets "$mets" --parameter $trainconfig $trainfilegrps
	java -Dfile.encoding=UTF-8 -Xmx3g -cp "$jar" "$main" --log-level "$LOG_LEVEL" \
		 -c train --mets "$mets" --parameter "$trainconfig" $trainfilegrps
}

# Run the evaluation over the `-ALIGN-` filegroups in the workspace
# directory of the given mets.xml file.  Usage:
# `ocrd-cis-run-evaluation config mets`.
# * config: path to the configuration file
# * mets:   path to the mets file
ocrd-cis-run-evaluation() {
	local config=$1
	local mets=$2
	local workspace=$(dirname "$mets")
	local main="de.lmu.cis.ocrd.cli.Main"
	local jar=$(cat "$config" | jq --raw-output '.jar')
	local evalconfig=$(cat "$config" | jq --raw-output '.evalparampath')

	# get -ALIGN- filegroups
	pushd "$workspace"
	local trainfilegrps=""
	for fg in $(ocrd workspace list-group); do
		if [[ $fg == *"-ALIGN-"* ]]; then
			trainfilegrps="$trainfilegrps -I $(basename $fg)"
		fi
	done
	popd
	# run evaluation
	for cmd in evaluate-dle evaluate-rrdm; do
		ocrd-cis-log java -Dfile.encoding=UTF-8 -Xmx3g -cp "$jar" "$main" -c "$cmd" \
			 --mets "$mets" \
			 --parameter "$param" \
			 $trainfilegrps \
			 --log-level $LOG_LEVEL
		java -Dfile.encoding=UTF-8 -Xmx3g -cp "$jar" "$main" -c "$cmd" \
			 --mets "$mets" \
			 --parameter "$evalconfig" \
			 $trainfilegrps \
			 --log-level $LOG_LEVEL
	done
}

# Download the ground truth archives and unzip them into a dedicated
# directory.  Usage: `ocrd-cis-download-and-extract-ground-truth url
# dir`.
# * url: URL of the archives
# * dir: output directory for the extracted archives
ocrd-cis-download-and-extract-ground-truth() {
	local url=$1
	local dir=$2
	mkdir -p "$dir"
	pushd "$dir"
	ocrd-cis-log "downloading $url"
	wget -r -np -l1 -nd -N -A zip -erobots=off "$url" || true # ignore exit status of wget
	for zip in *.zip; do
		# this archive is broken
		if [[ "$(basename $zip)" == $'bi\u00dfmarck_carmina_1657.zip' ]]; then continue; fi
		unzip -u -o $zip
	done
	popd
}
