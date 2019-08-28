#/bin/bash
set -e

TMP=$(mktemp -d)
WS="$TMP/ws"
trap "rm -rfv $TMP" EXIT
file=$(realpath $(dirname $0)/data/benner_herrnhuterey04_1748_0015.xml)

ocrd workspace init $WS
pushd $WS
ocrd workspace add \
	 -G OCR-D-CIS-TEST \
	 -i test01 \
	 -m 'application/vnd.prima.page+xml' \
	 "$file"
popd

# profile using mock profiler data/profiler
ocrd-cis-profile --log-level DEBUG \
				 -I OCR-D-CIS-TEST \
				 -O OCR-D-CIS-PROFILE \
				 -m $WS/mets.xml \
				 -p <(echo "{\"backend\":\"test\",\"executable\":\"$(dirname $0)/data/profiler\"}")

pushd $WS
if [[ ! -f $(ocrd workspace find -G OCR-D-CIS-PROFILE) ]]; then
	echo "cannot find profile in workspace"
	exit 1
fi
if [[ ! -f $(ocrd workspace find -m 'application/json+gzip') ]]; then
	echo "cannot find profile in workspace"
	exit 1
fi
