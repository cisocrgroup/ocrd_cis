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

#############################################
# shuffle files into eval and train folders #
#############################################
odirtrain="$odir/train-corpus"
odireval="$odir/eval-corpus"
if [[ ! -d "$odirtrain" ]]; then
	mkdir -p "$odirtrain" "$odireval"
	i=1
	for xml in $(find "$idir" -type f -name '*.xml' | sort); do
		img=$(ocrd-cis-find-image-for-xml "$(dirname $xml)" "$xml")
		x=$((i%2))
		if [[ $x == 0 ]]; then
			cp "$xml" "$img" "$odirtrain"
		else
			cp "$xml" "$img" "$odireval"
		fi
		i=$((i+1))
	done
fi

#########
# train #
#########
if [[ ! -d "$odir/trainws" ]]; then
	mkdir -p "$odir/trainws"
	pushd "$odir/trainws"
	ocrd workspace init .
	popd
fi
ocrd-cis-run-ocr-and-align "$config" "$odir/trainws/mets.xml" "$odirtrain" train-shuffle GT
ocrd-cis-run-training "$config" "$odir/trainws/mets.xml"

#########
# eval  #
#########
if [[ ! -d "$odir/evalws" ]]; then
	mkdir -p "$odir/evalws"
	pushd "$odir/evalws"
	ocrd workspace init .
	popd
fi
ocrd-cis-run-ocr-and-align "$config" "$odir/evalws/mets.xml" "$odireval" eval-shuffle GT
ocrd-cis-run-evaluation "$config" "$odir/evalws/mets.xml"
