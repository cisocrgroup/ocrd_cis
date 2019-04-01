#!/bin/bash
set -e
bdir=$(dirname "$0")
source "$bdir/ocrd-cis-lib.sh"

config=$(ocrd-cis-getopt -P --parameter $*)
odir=eval-ocrd-self
url=$(cat "$config" | jq --raw-output .gtlink)

ocrd-cis-download-and-extract-ground-truth "$url" downloads

#############################################
# shuffle files into eval and train folders #
#############################################
odirtrain="$odir/train-corpus"
odireval="$odir/eval-corpus"
mkdir -p "$odirtrain" "$odireval"
for dir in downloads/*; do
	if [[ ! -d "$dir" ]]; then continue; fi
	i=1
	for xml in $(find "$dir" -type f -name '*.xml' | grep -v 'alto' | shuf); do
		echo "xml: $xml"
		img=$(ocrd-cis-find-image-for-xml "$dir" "$xml")
		echo "xml: $xml img: $img"
		x=$((i%2))
		name=$(basename "$dir")
		if [[ $x == 0 ]]; then
			mkdir -p "$odireval/$name"
			cp "$xml" "$img" "$odireval/$name"
		else
			mkdir -p "$odirtrain/$name"
			cp "$xml" "$img" "$odirtrain/$name"
		fi
		i=$((i+1))
	done
done

#########
# train #
#########
mkdir -p "$odir/trainws"
pushd "$odir/trainws"
ocrd workspace init .
popd
for dir in "$odirtrain/"*; do
	name=$(basename "$dir")
	ocrd-cis-run-ocr-and-align "$config" "$odir/trainws/mets.xml" "$dir" "train-ocrd-self-$name" GT
done
ocrd-cis-run-training "$config" "$odir/trainws/mets.xml"

#########
# eval  #
#########
mkdir -p "$odir/evalws"
pushd "$odir/evalws"
ocrd workspace init .
popd
for dir in "$odireval/"*; do
	name=$(basename "$dir")
	ocrd-cis-run-ocr-and-align "$config" "$odir/evalws/mets.xml" "$dir" "eval-ocrd-self-$name" GT
done
ocrd-cis-run-evaluation "$config" "$odir/evalws/mets.xml"
