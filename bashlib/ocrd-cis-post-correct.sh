#!/bin/bash
# set -x
source $(dirname $0)/ocrd-cis-lib.sh

# Post correct OCR results.  Args:
# * the option --mets for the workspace.
# * the option --log-level is passed to other tools.
# * the option --parameter is used as configuration.
# * the option --output-file-grp defines the output filegroup of the
#   post-corrected page xml files.
# * the option --input-file-grp specifies the file group of the ocr
#   results that will be processed.

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
ocrd-cis-getopt -I --input-file-grp $* || ocrd-cis-fail "error: missing input file group (--input-file-grp)"
INPUT_FILE_GRP=$OCRD_CIS_OPTARG
ocrd-cis-debug "input file group: $INPUT_FILE_GRP"

#
# do additional ocrs and align
#
ocrd-cis-info "step: additional ocr and alignment"
# preset in case that there are no ocr steps
XML_INPUT_FILE_GRP="$INPUT_FILE_GRP"
XML_OUTPUT_FILE_GRP="$INPUT_FILE_GRP"
alignfgs="$XML_INPUT_FILE_GRP" # master ocr comes first
n=1
for cmd in $(cat $PARAMETER | jq -r '.ocrSteps[] | @base64'); do
	XML_OUTPUT_FILE_GRP="$XML_OUTPUT_FILE_GRP-OCR$n"
	n=$((n+1))
	cmd=$(echo $cmd | base64 -d)
	eval ocrd-cis-debug "$cmd"
	eval $cmd || exit 1
	alignfgs="$alignfgs,$XML_OUTPUT_FILE_GRP"
done
alignfg="$XML_INPUT_FILE_GRP-ALIGN"
ocrd-cis-debug ocrd-cis-align --mets $METS \
			   --input-file-grp "$alignfgs" \
			   --output-file-grp "$alignfg"
ocrd-cis-align --mets $METS \
			   --input-file-grp "$alignfgs" \
			   --output-file-grp "$alignfg"

#
# post correction
#
pcdir="$tmpdir/training"
mkdir -p "$pcdir"
main="de.lmu.cis.ocrd.cli.Main"
jar=$(ocrd-cis-data -jar)
nocr=$(jq ".ocrSteps | length+1" "$PARAMETER")
ocrd-cis-info "step: post-correction"
ocrd-cis-debug java -Dfile.encoding=UTF-8 -Xmx3g -cp $jar $main \
	 --log-level $LOG_LEVEL \
	 -c post-correct \
	 --mets $METS \
	 --parameter <(jq ".postCorrection.nOCR = \"$nocr\" | .postCorrection" "$PARAMETER") \
	 --input-file-grp "$trainfgs"
java -Dfile.encoding=UTF-8 -Xmx3g -cp $jar $main \
	 --log-level $LOG_LEVEL \
	 -c post-correct \
	 --mets $METS \
	 --parameter <(jq ".postCorrection.nOCR = \"$nocr\" | .postCorrection" "$PARAMETER") \
	 --input-file-grp "$alignfg" \
	 --output-file-grp "$OUTPUT_FILE_GRP"

#
# add protocols to the workspace
#
pushd $(dirname $METS)
if [[ -f "$pcdir/le-protocol.json" ]]; then
	ocrd-cis-info "step: add lexicon extension protocol"
	ocrd workspace add \
		 --file-grp "$OUTPUT_FILE_GRP-LE-PROTOCOL" \
	 --mimetype "application/json" \
	 --file-id "ocrd-cis-le-protocol.json" \
	 "$pcwdir/le-protocol.json"
fi
if [[ -f "$pcdir/dm-protocol.json" ]]; then
	ocrd-cis-info "step: add desicion maker protocol"
	ocrd workspace add \
		 --file-grp "$OUTPUT_FILE_GRP-DM-PROTOCOL" \
	 --mimetype "application/json" \
	 --file-id "ocrd-cis-dm-protocol.json" \
	 "$pcwdir/md-protocol.json"
fi
popd
