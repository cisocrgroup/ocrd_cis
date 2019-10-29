#!/bin/bash
# set -x
source $(dirname $0)/ocrd-cis-lib.sh

# Train a model. The tool somewhat mimics the behaviour of other ocrd
# tools:
# * the option --mets for the workspace.
# * the option --log-level is passed to other tools.
# * the option --parameter is used as configuration.
# * the option --output-file-grp defines the output filegroup of the
#   model file.
# * the option --input-file-grp is ignored.

#
# tmp dir and cleanup
#
tmpdir=$(mktemp -d)
trap "rm -rfv $tmpdir" EXIT

#
# command line arguments
#
LOG_LEVEL=INFO
if ocrd-cis-getopt -l --log-level $*; then
	LOG_LEVEL=$OCRD_CIS_OPTARG
fi
ocrd-cis-debug "log-level: $LOG_LEVEL"
ocrd-cis-getopt -m --mets $* || ocrd-cis-fail "error: missing METS file (--mets)"
METS=$OCRD_CIS_OPTARG
METS=$(realpath $METS)
ocrd-cis-debug "mets: $METS"
ocrd-cis-getopt -p --parameter $* || ocrd-cis-fail "error: missing configuration file (--parameter)"
PARAMETER=$OCRD_CIS_OPTARG
PARAMETER=$(realpath $PARAMETER)
ocrd-cis-debug "parameter: $PARAMETER"
ocrd-cis-getopt -O --output-file-grp $* || ocrd-cis-fail "error: missing output file group (--output-file-grp)"
OUTPUT_FILE_GRP=$OCRD_CIS_OPTARG
ocrd-cis-debug "output file group: $OUTPUT_FILE_GRP"

#
# download and prepare ground truth from archives
#
ocrd-cis-info "step: ground truth"
pushd $(dirname $METS)
for archive in $(cat $PARAMETER | jq -r '.gtArchives[]'); do
	archivedir="$tmpdir/$(basename $archive)"
	mkdir -p "$archivedir"
	if [[ $archive == http://* ]] || [[ $archive == https://* ]]; then
		ocrd-cis-download-and-add-gt-zip "$archive" "$archivedir"
	else
		ocrd-cis-add-gt-zip "$archive" "$archivedir"
	fi
	if [[ -z $xmlfgs ]]; then
		xmlfgs=$OCR_D_CIS_GT_FILEGRP
		imgfgs=$OCR_D_CIS_IMG_FILEGRP
	else
		xmlfgs="$xmlfgs $OCR_D_CIS_GT_FILEGRP"
		imgfgs="$imgfgs $OCR_D_CIS_IMG_FILEGRP"
	fi
done
popd

#
# do image pre-processing
#
ocrd-cis-info "step: image pre processing"
tmpimgfgs=""
tmpxmlfgs=""
for imgfg in $imgfgs; do
	xmlfg=${imgfg/IMG/GT}
	IMG_INPUT_FILE_GRP=$imgfg
	XML_INPUT_FILE_GRP=$xmlfg
	# preset in case that there are no image preprocessing steps
	IMG_OUTPUT_FILE_GRP=$imgfg
	XML_OUTPUT_FILE_GRP=$xmlfg
	n=1
	for cmd in $(cat $PARAMETER | jq -r '.imagePreprocessingSteps[] | @base64'); do
		IMG_OUTPUT_FILE_GRP="${imgfg/IMG/IMG-IPP$n}"
		XML_OUTPUT_FILE_GRP="${xmlfg/GT/XML-IPP$n}"

		n=$((n+1))
		cmd=$(echo $cmd | base64 -d)
		eval ocrd-cis-debug "$cmd"
		eval $cmd
		IMG_INPUT_FILE_GRP=$IMG_OUTPUT_FILE_GRP
		XML_INPUT_FILE_GRP=$XML_OUTPUT_FILE_GRP
	done
	if [[ -z tmpimgfgs ]]; then
		tmpimgfgs=$IMG_OUTPUT_FILE_GRP
		tmpxmlfgs=$XML_OUTPUT_FILE_GRP
	else
		tmpimgfgs="$tmpimgfgs $IMG_OUTPUT_FILE_GRP"
		tmpxmlfgs="$tmpxmlfgs $XML_OUTPUT_FILE_GRP"
	fi
done
imgfgs=$tmpimgfgs
xmlfgs=$tmpxmlfgs

#
# do the ocr and align
#
ocrd-cis-info "step: ocr and alignment"
for xmlfg in $xmlfgs; do
	# preset in case that there are no ocr steps
	XML_INPUT_FILE_GRP=$xmlfg
	XML_OUTPUT_FILE_GRP=$xmlfg
	alignfgs=""
	n=1
	for cmd in $(cat $PARAMETER | jq -r '.ocrSteps[] | @base64'); do
		XML_OUTPUT_FILE_GRP="${xmlfg/XML/OCR$n}"
		n=$((n+1))
		cmd=$(echo $cmd | base64 -d)
		eval ocrd-cis-debug "$cmd"
		eval $cmd || exit 1
		if [[ -z $alignfgs ]]; then
			alignfgs="$XML_OUTPUT_FILE_GRP"
		else
			alignfgs="$alignfgs,$XML_OUTPUT_FILE_GRP"
		fi
	done
	alignfgs="$alignfgs,$xmlfg" # append gt filegroup
	trainfg="${xmlfg/XML/ALIGN}"
	ocrd-cis-align --mets $METS \
				   --input-file-grp "$alignfgs" \
				   --output-file-grp "$trainfg"
	werfg="${xmlfg/XML/WER}"
	ocrd-cis-wer --mets $METS \
				 --input-file-grp "$trainfg" \
				 --output-file-grp "$werfg"
	# sadly we cannot use something like ocrd workspace find -G | grep ALIGN
	if [[ -z $trainfgs ]]; then
		trainfgs="$trainfg"
	else
		trainfgs="$trainfg,$trainfgs"
	fi
done

#
# training
#
traindir="$tmpdir/training"
mkdir -p "$traindir"
main="de.lmu.cis.ocrd.cli.Main"
nocr=$(jq ".ocrSteps | length" "$PARAMETER")
ocrd-cis-info "step: training"
# eval ocrd-cis-debug java -Dfile.encoding=UTF-8 -Xmx3g -cp $(ocrd-cis-data -jar) $main \
# 	 --log-level $LOG_LEVEL \
# 	 -c train \
# 	 --mets $METS \
# 	 --parameter <(jq ".training.dir = \"$traindir\"" "$PARAMETER") \
# 	 --input-file-grp "$trainfgs"
java -Dfile.encoding=UTF-8 -Xmx3g -cp $(ocrd-cis-data -jar) $main \
	 --log-level $LOG_LEVEL \
	 -c train \
	 --mets $METS \
	 --parameter <(jq ".training.dir = \"$traindir\" | .training.nOCR = \"$nocr\" | .training" "$PARAMETER") \
	 --input-file-grp "$trainfgs"

#
# add model and training resources to workspace
#
pushd $(dirname $METS)
ocrd-cis-info "step: cleanup"
ocrd workspace add \
	 --file-grp "$OUTPUT_FILE_GRP" \
	 --mimetype "application/zip" \
	 --file-id "ocrd-cis-model.zip" \
	 "$traindir/model.zip"
rm -rf "$traindir/model.zip"

zip -r "$tmpdir/training.zip" "$traindir"
ocrd workspace add \
	 --file-grp "$OUTPUT_FILE_GRP-TRAINING" \
	 --mimetype "application/zip" \
	 --file-id "ocrd-cis-training.zip" \
	 "$tmpdir/training.zip"
popd
