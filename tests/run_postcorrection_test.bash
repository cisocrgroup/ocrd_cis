#!/bin/bash
set -e
source $(dirname $0)/test_lib.bash

ocrd_cis_init_ws blumenbach_anatomie_1805.ocrd.zip
# test if there are 3 gt files
pushd "$tmpws"
found_files=0
for file in $(ocrd workspace find -G OCR-D-GT-SEG-LINE); do
	if [[ ! -f "$file" ]]; then
		echo "cannot find ground truth file: $file"
		exit 1
	fi
	found_files=$((found_files + 1))
done
if [[ $found_files != 3 ]]; then
	echo "invalid number of files: $found_files"
	exit 1
fi
popd

ocrd_cis_align

mkdir "$tmpdir/bin"
cat > "$tmpdir/bin/profiler.bash" <<EOF
#!/bin/bash
cat > /dev/null
echo '{}'
EOF
chmod a+x "$tmpdir/bin/profiler.bash"
ocrd-cis-postcorrect --log-level DEBUG \
					 -I OCR-D-CIS-ALIGN \
					 -O OCR-D-CIS-POSTCORRECT \
					 -m $tmpws/mets.xml \
					 --parameter <(cat <<EOF
{
"profilerPath": "$tmpdir/bin/profiler.bash",
"profilerConfig": "ignored",
"model": "$(ocrd-cis-data -model)",
"nOCR": 2
}
EOF
)

pushd $tmpws
found_files=0
for file in $(ocrd workspace find -G OCR-D-CIS-POSTCORRECT); do
	if [[ ! -f "$file" ]]; then
		echo "$file: not a file"
		exit 1
	fi
	found_files=$((found_files + 1))
done
if [[ $found_files != 3 ]]; then
	echo "invalid number of files: $found_files"
	exit 1
fi
popd
