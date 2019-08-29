#/bin/bash
set -e

tmp=$(mktemp -d)
tmpws="$tmp/ws"
trap "rm -rfv $tmp" EXIT
pagexmlfile=$(realpath $(dirname $0)/data/benner_herrnhuterey04_1748_0015.xml)

ocrd workspace init $tmpws
