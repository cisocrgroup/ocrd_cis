#!/usr/bin/bash

TMP=$(mktemp -d '/tmp/ocrd-cis-csv.XXXXXXXXXX')
trap "rm -vrf $TMP" EXIT

if [[ -z $1 ]]; then
	echo "usage: $0 <result-dir>"
	exit 1
fi

function csv() {
	local what=$1
	local file=$2
	local name=$(basename $file)

	local tmp=$TMP/$what-$name.csv
	tmp=${tmp/.txt/}
	rm -f $tmp
	touch $tmp
	# Multiple awk calls to preserve a specific order
	# corr
	cat $file | awk -F ': ' \
					-e '/number of ocr correct tokens/{print "corr,"$2;}' \
					>> $tmp
	# incorr
	cat $file | awk -F ': ' \
					-e '/number of ocr error tokens/{print "incorr,"$2;}' \
					>> $tmp
	# not in top 5
	cat $file | awk -F ': ' \
					-e '/number of no good placement available/{print "not in top 5,"$2;}' \
					>> $tmp
	# Top 1 prof
	cat $file | awk -F ': ' \
					-e '/number of good profiler suggestions/{print "Top 1 prof,"$2;}' \
					>> $tmp
	# Top 1 ranker
	cat $file | awk -F ': ' \
					-e '/number of good placements/{print "Top 1 ranker,"$2;}' \
					>> $tmp
	# succ.c
	cat $file | awk -F ': ' \
					-e '/real improvements/{print "succ.c,"$2;}' \
					>> $tmp
	# lostch
	cat $file | awk -F ': ' \
					-e '/^missed opportunities/{print "lostch,"$2;}' \
					>> $tmp
	# inf.c
	cat $file | awk -F ': ' \
					-e '/^disimprovements/{print "inf.c,"$2;}' \
					>> $tmp
	# OCR c
	cat $file | awk -F ': ' \
					-e '/number of correct OCR tokens \(after correction\)/{print "OCR c,"$2;}' \
					>> $tmp
	# errors (type i, ii, iii, iv)
	cat $file | awk -F ': ' \
					-e '/type\(i\)/{gsub(/ /,"",$2); print "type I,"$2;}' \
					>> $tmp
	cat $file | awk -F ': ' \
					-e '/type\(ii\)/{gsub(/ /,"",$2); print "type II,"$2;}' \
					>> $tmp
	cat $file | awk -F ': ' \
					-e '/type\(iii\)/{gsub(/ /,"",$2); print "type III,"$2;}' \
					>> $tmp
	cat $file | awk -F ': ' \
					-e '/type\(iv\)/{gsub(/ /,"",$2); print "type IV,"$2;}' \
					>> $tmp
}

for sdir in "$1"*; do
	if [[ ! -d "$sdir" ]]; then
		continue
	fi
	for rfile in $sdir/train/dm-result_*.txt; do
		csv $(basename $sdir) $rfile
	done
done
# for tfile in $TMP/*.csv; do
# 	echo $tfile
# 	cat $tfile;
# done
paste -d,\
	  $TMP/eval-bodenstein-ocrd-dm-result_no_dle_1.csv\
	  $TMP/eval-bodenstein-ocrd-dm-result_no_dle_2.csv\
	  $TMP/eval-bodenstein-ocrd-dm-result_1.csv\
	  $TMP/eval-bodenstein-ocrd-dm-result_2.csv\
	  $TMP/eval-bodenstein-shuffle-dm-result_no_dle_1.csv\
	  $TMP/eval-bodenstein-shuffle-dm-result_no_dle_2.csv\
	  $TMP/eval-bodenstein-shuffle-dm-result_1.csv\
	  $TMP/eval-bodenstein-shuffle-dm-result_2.csv\
	  $TMP/eval-grenzboten-ocrd-dm-result_no_dle_1.csv\
	  $TMP/eval-grenzboten-ocrd-dm-result_no_dle_2.csv\
	  $TMP/eval-grenzboten-ocrd-dm-result_1.csv\
	  $TMP/eval-grenzboten-ocrd-dm-result_2.csv\
	  $TMP/eval-grenzboten-shuffle-dm-result_no_dle_1.csv\
	  $TMP/eval-grenzboten-shuffle-dm-result_no_dle_2.csv\
	  $TMP/eval-grenzboten-shuffle-dm-result_1.csv\
	  $TMP/eval-grenzboten-shuffle-dm-result_2.csv\
	  | awk -F, -e '{print $1","$2","$4","$6","$8","$10","$12","$14","$16","$18","$20","$22","$24","$26","$28","$30","$32;'}
