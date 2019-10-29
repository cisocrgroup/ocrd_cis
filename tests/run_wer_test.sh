#!/bin/bash

source $(dirname $0)/test_lib.sh

pushd "$tmpws"
ocrd workspace add \
	 --file-grp "WERFILEGRP" \
	 --mimetype "application/vnd.prima.page+xml" \
	 --file-id "WERFILEID" \
	 "$werfile"
popd

ocrd-cis-wer \
	--mets "$tmpws/mets.xml" \
	--output-file-grp "WER" \
	--input-file-grp "WERFILEGRP"


# tests
pushd "$tmpws"
if [[ ! -f $(ocrd workspace find -G "WER") ]]; then
   	echo "missing wer file"
	exit 1
fi
if [[ $(jq '.totalWords' $(ocrd workspace find -G "WER")) != 3 ]]; then
   	echo "invalid number of words"
	exit 1
fi
if [[ $(jq '.incorrectWords' $(ocrd workspace find -G "WER")) != 2 ]]; then
   	echo "invalid number of bad words"
	exit 1
fi
if [[ $(jq '.correctWords' $(ocrd workspace find -G "WER")) != 1 ]]; then
   	echo "invalid number of good words"
	exit 1
fi
popd
