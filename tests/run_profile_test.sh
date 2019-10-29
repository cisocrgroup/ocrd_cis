#!/bin/bash
source $(dirname $0)/test_lib.sh

# add page xml file
pushd $tmpws
ocrd workspace add \
	 -G OCR-D-CIS-TEST \
	 -i test01 \
	 -m 'application/vnd.prima.page+xml' \
	 "$pagexmlfile"
popd

profiler=$(realpath $(dirname $0)/data/profiler)
if [[ ! -f $profiler ]]; then
	echo "cannot find $profiler"
	exit 1
fi

# profile using mock profiler at data/profiler
ocrd-cis-profile --log-level DEBUG \
				 -I OCR-D-CIS-TEST \
				 -O OCR-D-CIS-PROFILE \
				 -m $tmpws/mets.xml \
				 -p <(echo "{\"backend\":\"test\",\"executable\":\"$profiler\"}")

pushd $tmpws
if [[ ! -f $(ocrd workspace find -G OCR-D-CIS-PROFILE) ]]; then
	echo "cannot find profile file group in workspace"
	exit 1
fi
if [[ ! -f $(ocrd workspace find -m 'application/json+gzip') ]]; then
	echo "cannot find profile in workspace"
	exit 1
fi
