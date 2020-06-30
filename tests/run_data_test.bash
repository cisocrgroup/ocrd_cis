#!/bin/bash
set -e

if [[ ! -f $(ocrd-cis-data -jar) ]] ; then
	echo "jar file does not exist";
	exit 1
fi

if [[ ! -f $(ocrd-cis-data -3gs) ]] ; then
	echo "three grams file does not exist";
	exit 1
fi

if [[ ! -f $(ocrd-cis-data -config) ]] ; then
	echo "config file does not exist";
	exit 1
fi

if [[ ! -f $(ocrd-cis-data -model) ]] ; then
	echo "model file does not exist";
	exit 1
fi
