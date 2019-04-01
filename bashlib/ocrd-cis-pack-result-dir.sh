#!/bin/bash
set -e

# bdir=$(dirname "$0")
# source "$bdir/ocrd-cis-lib.sh"

idir=$1
odir=$2
tar=$(basename "$idir")
tar="$odir/$tar.tar.bz2"

GZIP=-9 # use best compression
pushd $idir
tar -cjf "$tar" **/train/*.txt
tar -tf "$tar"
echo $tar
popd
