#!/bin/bash

set -e
# set -x

source ocrd-cis.sh

config=$(ocrd-cis-getopt -P --parameter $*)
workspace=$(cat $config | jq --raw-output '.workspace')
LOG_LEVEL=DEBUG
gtlink=$(cat $config | jq --raw-output '.gtlink')

#########################
# download ground truth #
#########################
mkdir -p downloads
pushd downloads
echo "downloading $gtlink"
wget -r -np -l1 -nd -N -A zip -erobots=off $gtlink || true # ignore exit status of wget
popd

##################
# unzip archives #
##################
pushd downloads
for zip in *.zip; do
	unzip -u -o $zip
done
popd

####################
# create workspace #
####################
if [[ ! -d $workspace ]]; then
	mkdir -p $workspace
	pushd $workspace
	ocrd $workspace init .
	popd
fi

##########################
# download the ocrd.jar  #
##########################
ocrd-cis-download-jar $(cat $config | jq --raw-output '.jar')

################################
# add pagexml files and images #
################################
max=-1 # set to -1 for all
for dir in downloads/*; do
	if [[ max -eq 0 ]]; then
		break;
	fi
	max=$((max-1))
	if [[ -d $dir ]]; then
		basefilegrp=$(basename $dir)
		for tif in $(find $dir -type f -name '*.tif'); do
			# find according page xml file
			echo $tif
			tifdir=$(/usr/bin/dirname $tif)
			name=$(basename $tif)
			name=${name/.tif/.xml}
			xml="$tifdir/page/$name"
			# handle cases where image names are like 0001.tif, 0002.tif ...
			# and page xml names are like nn_zeitung_XXXX_0001.xml ...
			if [[ ! -f $xml ]]; then
				echo $xml does not exist
				tmp=$(find $tifdir/page -type f -iname "*$name")
				if [[ ! -f $tmp ]]; then
					echo "cannot find according page xml for $tif"
					exit 1
				fi
				xml=$tmp
			fi
			# add tif and page xml files to workspace
			echo $tif $xml
			pushd $workspace
			fileid=$(basename $tif)
			filegrp="OCR-D-IMG-TRAIN-$basefilegrp"
			mimetype="img/tif"
			tifpath="$filegrp/$fileid"
			ocrd workspace add --file-grp $filegrp --file-id $fileid --mimetype $mimetype ../$tif
			fileid=$(basename $xml)
			filegrp="OCR-D-GT-TRAIN-$basefilegrp"
			mimetype="application/vnd.prima.page+xml"
			fixpath="$filegrp/$fileid"
			ocrd workspace add --file-grp $filegrp --file-id $fileid --mimetype $mimetype ../$xml
			# imageFilepath in page xml
			abstifpath=$(realpath $tifpath)
			sed -i $fixpath -e "s#imageFilename=\"[^\"]*\"#imageFilename=\"$abstifpath\"#"
			popd
		done
	fi
done

#########################################################
# run ocr, alignment and profiler on the training files #
#########################################################
for dir in $workspace/*-GT-TRAIN-*; do
	alignifilegrps=""
	ifilegrp=$(basename $dir)
	# run ocrs
	for i in $(seq 0 $(cat $config | jq ".ocr | length-1")); do
		type=$(cat $config | jq --raw-output ".ocr[$i].type")
		path=$(cat $config | jq --raw-output ".ocr[$i].path")
		utype=$(echo $type | tr '[:lower:]' '[:upper:]')
		ofilegrp=${ifilegrp/-GT-TRAIN-/-OCR-TRAIN-$utype-$((i+1))-}
		case $type in
			"ocropy")
				alignifilegrps="$alignifilegrps $ofilegrp"
				ocrd-cis-ocropy-recognize \
					--input-file-grp $ifilegrp \
					--output-file-grp $ofilegrp \
					--mets $workspace/mets.xml \
					--parameter $path \
					--log-level $LOG_LEVEL
				;;
			# "tesseract")
			# 	echo "tesseract $path"
			# 	;;
			*)
				echo "invalid ocr type: $type"
				exit 1
				;;
		esac
	done
	# align ocrs and ground truth
	alignifilegrps="$alignifilegrps $ifilegrp"
	alignifilegrps=$(ocrd-cis-join-by , $alignifilegrps)
	alignofilegrp=${ifilegrp/-GT-TRAIN-/-ALIGN-TRAIN-}
	echo $alignifilegrps
    ocrd-cis-align \
        --input-file-grp $alignifilegrps \
        --output-file-grp $alignofilegrp \
        --mets $workspace/mets.xml \
        --parameter $(cat $config | jq --raw-output ".alignparampath") \
        --log-level $LOG_LEVEL
done

###################################
# train automatic postcorrection  #
###################################
trainfilegrps=""
for d in $workspace/*; do
	if [[ -d $d ]]; then
		if [[ $d == *"-ALIGN-TRAIN-"* ]]; then
			filegrp=$(basename $d);
			trainfilegrps="$trainfilegrps -I $filegrp";
		fi
	fi
done
main="de.lmu.cis.ocrd.cli.Main"
jar=$(cat "$config" | jq --raw-output '.jar')
echo java -Dfile.encoding=UTF-8 -Xmx3g -cp $jar $main --log-level $LOG_LEVEL \
	 -c train --mets $workspace/mets.xml --parameter $trainconfig $trainfilegrps
java -Dfile.encoding=UTF-8 -Xmx3g -cp $jar $main --log-level $LOG_LEVEL \
	 -c train --mets $workspace/mets.xml --parameter $trainconfig $trainfilegrps
