#!/bin/bash

set -e

source ocrd-cis.sh
ocrd-cis-download-jar config/ocrd.jar
config=$(ocrd-cis-getopt -P --parameter $*) || echo "missing --parameter|-P flag"
main=$(cat $config |jq --raw-output '.main')
jar=$(cat $config | jq --raw-output '.jar')
java -Dfile.encoding=UTF-8 -Xmx3g -cp $jar $main -c profile $*
