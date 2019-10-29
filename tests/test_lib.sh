#/bin/bash
set -e

tmpdir=$(mktemp -d)
tmpws="$tmpdir/ws"
trap "rm -rfv $tmpdir" EXIT
pagexmlfile=$(realpath $(dirname $0)/data/benner_herrnhuterey04_1748_0015.xml)
werfile=$(realpath $(dirname $0)/data/benner_herrnhuterey04_1748_0015-wer.xml)

ocrd workspace init $tmpws
