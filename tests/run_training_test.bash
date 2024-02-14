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

# fix ocr for some entries (otherwise the training will fail)
pushd $tmpws
for f in $(ocrd workspace find -G OCR-D-CIS-ALIGN); do
	sed -i -e 's#<pc:Unicode>e.</pc:Unicode>#<pc:Unicode>Säugethiere.</pc:Unicode>#' $f
	sed -i -e 's#<pc:Unicode>E</pc:Unicode>#<pc:Unicode>Säugethieren</pc:Unicode>#' $f
done
popd

mkdir "$tmpdir/bin"
cat > "$tmpdir/bin/profiler.bash" <<EOF
#!/bin/bash
cat > /dev/null
echo '{"Säugethiere":{
"Candidates": [{
"Suggestion": "Säugethiere",
"Modern": "Säugetiere",
"Dict": "dict_modern_hypothetic_errors",
"HistPatterns": [{"Left":"t","Right":"th","Pos":5}],
"OCRPatterns": [],
"Distance": 0,
"Weight": 1.0
}]}}'
EOF
chmod a+x "$tmpdir/bin/profiler.bash"

java -jar $(ocrd-cis-data -jar) -c train \
	 --log-level DEBUG \
	 -I OCR-D-CIS-ALIGN \
	 -m $tmpws/mets.xml \
	 --parameter <(
cat $(ocrd-cis-data -config) \
	| sed -e "s#/path/to/profiler#$tmpdir/bin/profiler.bash#" \
	| sed -e "s#/path/to/trigrams.csv#$(ocrd-cis-data -3gs)#" \
	| sed -e "s#/path/to/train.dir#$tmpdir/train#"
)

[[ -f "$tmpdir/train/model.zip" ]] || fail "$tmpdir/train/model.zip not found"
