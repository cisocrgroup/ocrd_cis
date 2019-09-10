#!/bin/bash
source $(dirname $0)/test_lib.sh

other1=$tmpdir/other1.xml
cat $pagexmlfile \
	| sed -e 's/Theil/Teyl/g' \
		  -e 's/deſen/defen/g' \
		  -e 's/Philadelphia/Philadclplia/g' \
	> $other1

other2=$tmpdir/other2.xml
cat $pagexmlfile | sed -e 's/ſ/f/g' > $other2

# add page xml files to align
pushd $tmpws
ocrd workspace add \
	 -G OCR-D-CIS-TEST-1 \
	 -i test01 \
	 -m 'application/vnd.prima.page+xml' \
	 "$pagexmlfile"
ocrd workspace add \
	 -G OCR-D-CIS-TEST-2 \
	 -i test02 \
	 -m 'application/vnd.prima.page+xml' \
	 "$other1"
ocrd workspace add \
	 -G OCR-D-CIS-TEST-3 \
	 -i test03 \
	 -m 'application/vnd.prima.page+xml' \
	 "$other2"
popd

# align the three workspaces
ocrd-cis-align --log-level DEBUG \
			   -I OCR-D-CIS-TEST-1,OCR-D-CIS-TEST-2,OCR-D-CIS-TEST-3 \
			   -O OCR-D-CIS-ALIGN \
			   -m $tmpws/mets.xml

pushd $tmpws
if [[ ! -f $(ocrd workspace find -G OCR-D-CIS-ALIGN) ]]; then
	echo "cannot find aligned file group workspace"
	exit 1
fi
