#!/bin/bash

if [[ ! -f $(ocrd-cis-data -jar) ]] ; then
	echo "jar file does not exist";
	exit 1
fi

if [[ ! -f $(ocrd-cis-data -3gs) ]] ; then
	echo "three grams file does not exist";
	exit 1
fi
