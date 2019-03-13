#!/bin/bash

set -e
# set -x

source "$(dirname $0)/ocrd-cis-lib.sh"

config=$(ocrd-cis-getopt -P --parameter $*)
workspace=$(cat $config | jq --raw-output '.workspace')
mets="$workspace/mets.xml"
LOG_LEVEL=DEBUG
gtlink=$(cat $config | jq --raw-output '.gtlink')

function prepare() {
	ocrd-cis-download-and-extract-ground-truth "$gtlink" downloads

	####################
	# create workspace #
	####################
	if [[ ! -d $workspace ]]; then
		mkdir -p $workspace
		pushd $workspace
		ocrd workspace init .
		popd
	fi

	##########################
	# download the ocrd.jar  #
	##########################
	ocrd-cis-download-jar $(cat $config | jq --raw-output '.jar')

	###########################
	# run ocrs and align them #
	###########################
	max=-1 # set to -1 for all
	for dir in downloads/*; do
		if [[ max -eq 0 ]]; then
			break;
		fi
		if [[ -d $dir ]]; then
			max=$((max-1))
			ocrd-cis-run-ocr-and-align "$config" "$mets" "$dir" $(basename "$dir") "GT"
		fi
	done
}

if [[ ! -d "$workspace" ]]; then
	prepare
fi

###################################
# train automatic postcorrection  #
###################################
trainfilegrps=""
for d in $workspace/*; do
	if [[ -d $d ]]; then
		if [[ $d == *"-ALIGN-"* ]]; then
			filegrp=$(basename $d);
			trainfilegrps="$trainfilegrps -I $filegrp";
		fi
	fi
done

main="de.lmu.cis.ocrd.cli.Main"
jar=$(cat "$config" | jq --raw-output '.jar')
trainconfig=$(cat "$config" | jq --raw-output '.trainparampath')
ocrd-cis-log java -Dfile.encoding=UTF-8 -Xmx3g -cp $jar $main --log-level $LOG_LEVEL \
			 -c train --mets $workspace/mets.xml --parameter $trainconfig $trainfilegrps
java -Dfile.encoding=UTF-8 -Xmx3g -cp $jar $main --log-level $LOG_LEVEL \
	 -c train --mets $workspace/mets.xml --parameter $trainconfig $trainfilegrps
