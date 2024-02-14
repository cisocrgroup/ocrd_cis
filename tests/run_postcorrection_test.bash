#!/bin/bash
set -e
source $(dirname $0)/test_lib.bash

ocrd_cis_init_ws blumenbach_anatomie_1805.ocrd.zip
# test if there are 3 gt files
pushd "$tmpws"
found_files=0
for file in $(ocrd workspace find -G $OCRD_CIS_FILEGRP); do
	[[ -f "$file" ]] || fail "cannot find ground truth file: $file"
	found_files=$((found_files + 1))
done
(( found_files == 3 )) || fail "invalid number of files: $found_files"
popd

ocrd_cis_align

mkdir "$tmpdir/bin"
cat > "$tmpdir/bin/profiler.bash" <<EOF
#!/bin/bash
cat > /dev/null
echo '{}'
EOF
chmod a+x "$tmpdir/bin/profiler.bash"
ocrd-cis-postcorrect -l DEBUG \
			-I OCR-D-CIS-ALIGN \
			-O OCR-D-CIS-POSTCORRECT \
			-m $tmpws/mets.xml \
			-P profilerPath $tmpdir/bin/profiler.bash \
			-P profilerConfig ignored \
			-P model "$(ocrd-cis-data -model)" \
			-P nOCR 2

pushd $tmpws
found_files=0
for file in $(ocrd workspace find -G OCR-D-CIS-POSTCORRECT); do
	[[ -f "$file" ]] || fail "$file: not a file"
	found_files=$((found_files + 1))
done
(( found_files == 3 )) || fail "invalid number of files: $found_files"
popd
