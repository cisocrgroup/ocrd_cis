#!/bin/bash

set -e

idir=$1
odir=$2
n=$3
if [[ -z "$n" ]]; then n=25; fi

files=""
i=1
j=1
for f in $idir/*.gt.txt; do
	if [[ -z "$files" ]]; then
		files=$f
	else
		files="$files $f"
	fi
	if [[ $((i%n)) == 0 ]]; then
		mkdir -p "$odir"
		out=$(printf "$odir/%04d" $j)
		gocrd synpage -o "$out" $files
		files=""
		j=$((j+1))
	fi
	i=$((i+1))
done
if [[ ! -z "$files" ]]; then
	mkdir -p "$odir"
	out=$(printf "$odir/%04d" $j)
	gocrd synpage -o "$out" $files
	files=""
fi
