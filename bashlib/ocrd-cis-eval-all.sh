#!/bin/bash
set -e

bdir=$(dirname "$0")
source "$bdir/ocrd-cis-lib.sh"

get_eval_dir() {
	case $1 in
		bodenstein) echo "eval/1557-Bodenstein-WieSichMeniglich";;
		grenzboten) echo "eval/1841-DieGrenzboten";;
		*) exit 1;;
	esac
}

odir=$(date +%Y%m%d_%H_%M)
odir="results/$odir"

for how in shuffle ocrd; do
	for corpus in grenzboten bodenstein; do
		dir="eval-$corpus-$how"
		cmd="$bdir/ocrd-cis-eval-$how.sh"
		# rm -rf "$dir"
		ocrd-cis-log ./"$cmd" -P "config/config-$how-$corpus.json" $(get_eval_dir $corpus) "$dir"
		./"$cmd" -P "config/config-$how-$corpus.json" $(get_eval_dir $corpus) "$dir"

		mkdir -p "$odir"
		cp -r "$dir" "$odir"
	done
done
