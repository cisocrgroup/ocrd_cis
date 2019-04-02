#!/bin/bash
set -e
bdir=$(dirname "$0")
source "$bdir/ocrd-cis-lib.sh"

if [[ $# != 4 ]]; then
	echo "usage: $0 -P|--parameter config input-dir output-dir-basename"
	exit 2
fi
config=$(ocrd-cis-getopt -P --parameter $*)
idir=$3
odir=$4

##########################################
# train post correction from ocrd corpus #
##########################################
if [[ ! -d "$odir/trainws" ]]; then
	mkdir -p "$odir/trainws"
	pushd "$odir/trainws"
	ocrd workspace init .
	popd
fi

gtlink=$(cat "$config" | jq --raw-output '.gtlink')
ocrd-cis-download-and-extract-ground-truth "$gtlink" downloads
for dir in downloads/*; do
	if [[ ! -d "$dir" ]]; then continue; fi
	name=$(basename "$dir")
	ocrd-cis-run-ocr-and-align "$config" "$odir/trainws/mets.xml" "$dir" "train-ocrd-corpus-$name" GT
done
ocrd-cis-run-training "$config" "$odir/trainws/mets.xml"

#############
# evaluate  #
#############
mkdir -p "$odir/evalws"
pushd "$odir/evalws"
ocrd workspace init .
popd
ocrd-cis-run-ocr-and-align "$config" "$odir/evalws/mets.xml" "$idir" eval-ocrd-corpus GT
ocrd-cis-run-evaluation "$config" "$odir/evalws/mets.xml"
