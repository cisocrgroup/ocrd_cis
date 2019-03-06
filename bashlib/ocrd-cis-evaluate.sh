#!/bin/bash

set -e

source "$(dirname $0)/ocrd-cis-lib.sh"

config=$(ocrd-cis-getopt -P --parameter $*)
ifg=$(ocrd-cis-getopt -I --input-file-grp $*)
mets=$(ocrd-cis-getopt -M --mets $*)
workspace=$(/usr/bin/dirname "$mets")
jar=$(cat "$config" | jq --raw-output '.jar')
LOG_LEVEL=DEBUG

echo ifg $ifg
echo mets $mets
echo workspace $workspace
echo jar $jar
echo config $config

########################
# download newest jar  #
########################
ocrd-cis-download-jar "$jar"

##############################
# create workspace (really?) #
##############################
if [[ ! -d "$workspace" ]]; then
	mkdir -p "$workspace"
	pushd "$workspace"
	ocrd workspace init .
	popd
fi

###########################################
# search for intput file group directory  #
###########################################
inputdir=$(find . -type d -name "$ifg")
if [[ -z $inputdir ]]; then
	echo "cannot find input directory for $ifg"
	exit 1
fi

#######################################
# add gt and image files to worksapce #
#######################################
max=-1 # set to -1 for all
for pxml in $(find "$inputdir" -type f -name '*.xml'); do
	if [[ max -eq 0 ]]; then
		break;
	fi
	max=$((max-1))
	img=$(ocrd-cis-find-image-for-xml "$inputdir" $(basename "$pxml"))
	if [[ -z $img ]]; then
		echo cannot find image file for $img
		exit 1
	fi
	if [[ -f $img ]]; then
		ocrd-cis-add-pagexml-and-image-to-workspace \
			"$workspace" "OCR-D-GT-EVAL-$ifg" "$pxml" "OCR-D-IMG-EVAL-$ifg" "$img"
	fi
done

echo $OCRFILEGRPS
ocrd-cis-run-ocr "$config" "$mets" "OCR-D-GT-EVAL-$ifg" "OCR-D-OCR-EVAL-XXX-$ifg"
echo $OCRFILEGRPS
OCRFILEGRPS="$OCRFILEGRPS OCR-D-GT-EVAL-$ifg"
echo $OCRFILEGRPS
OCRFILEGRPS=$(ocrd-cis-join-by , $OCRFILEGRPS)
echo $OCRFILEGRPS
ocrd-cis-log ocrd-cis-align \
    --input-file-grp "$OCRFILEGRPS" \
    --output-file-grp "OCR-D-ALIGN-EVAL-$ifg" \
    --mets "$mets"\
    --parameter $(cat "$config" | jq --raw-output ".alignparampath") \
    --log-level $LOG_LEVEL

ocrd-cis-align \
    --input-file-grp "$OCRFILEGRPS" \
    --output-file-grp "OCR-D-ALIGN-EVAL-$ifg" \
    --mets "$mets" \
    --parameter $(cat "$config" | jq --raw-output ".alignparampath") \
    --log-level $LOG_LEVEL

main="de.lmu.cis.ocrd.cli.Main"
param=$(cat "$config" | jq --raw-output '.trainparampath')
for cmd in evaluate-dle evaluate-rrdm; do
	java -Dfile.encoding=UTF-8 -Xmx3g -cp "$jar" "$main" -c $cmd \
		 --mets "$mets" \
		 --parameter "$param" \
		 --input-file-grp "OCR-D-ALIGN-EVAL-$ifg" \
		 --log-level $LOG_LEVEL
done
