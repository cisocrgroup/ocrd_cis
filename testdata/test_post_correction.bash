#!/bin/bash
wdir=$(mktemp -d /tmp/1841-bodenstein-XXX)
profiler="$PWD/testdata/profiler.bash"
trap "rm -rf $wdir" EXIT

# Copy base files into the workspace and set it up.
cp testdata/179481.png $wdir
cp testdata/179481.xml $wdir
cd $wdir 
ocrd workspace --directory $wdir init 
ocrd workspace add 179481.png -G IMG -g 179481 -i IMG_179481
ocrd workspace add 179481.xml -G ALG -g 179481 -i ALG_179481

# Test the alignment.
ocrd-cis-align  --log-level DEBUG -I ALG,ALG,ALG -O TEST -p '{}'
file=$(ocrd workspace find -G TEST)
if [[ ! -f $file ]]; then 
    echo "cannot find $file"
    exit 1
fi

# Test the post-correction.
ocrd-cis-post-correct \
    -I ALG -O COR -p $(ocrd-cis-data -config) \
    -P model $(ocrd-cis-data -19th) -P profilerBin $profiler
file=$(ocrd workspace find -G COR)
if [[ ! -f $file ]]; then 
    echo "cannot find $file"
    exit 1
fi
