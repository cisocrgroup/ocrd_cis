#!/bin/bash

TMP_LOG=$(mktemp /tmp/cis-ocrd-run-test-XXXXXX.log)
function rmtl() {
		echo removing $TMP_LOG
		rm -f $TMP_LOG
}
trap rmtl EXIT

verbose="false"
while test $# -gt 0; do
		case $1 in
				--verbose|-v)
						verbose="true"
						shift
						;;
				*)
						break
						;;
		esac
done

for t in $@; do
		if test $verbose = "true"; then
				$t
				if test $? -ne 0; then
						exit 1
				fi
				echo $t OK
		else
				$t &> $TMP_LOG
				if test $? -ne 0; then
						cat $TMP_LOG
						exit 1
				fi
				echo $t OK
		fi
done
