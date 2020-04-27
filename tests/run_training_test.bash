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

# download ocr models
wget -P "$tmpdir/download" "http://cis.lmu.de/~finkf/fraktur1-00085000.pyrnn.gz"
wget -P "$tmpdir/download" "http://cis.lmu.de/~finkf/fraktur2-00062000.pyrnn.gz"
# run ocr
ocrd-cis-ocropy-recognize --log-level DEBUG \
						  --input-file-grp "OCR-D-GT-SEG-LINE" \
						  --output-file-grp OCR-D-CIS-OCR-1 \
						  --mets "$tmpws/mets.xml" \
						  --parameter <(cat <<EOF
{
	"textequiv_level": "word",
	"model": "$tmpdir/download/fraktur1-00085000.pyrnn.gz"
}
EOF
)
ocrd-cis-ocropy-recognize --log-level DEBUG \
						  --input-file-grp "OCR-D-GT-SEG-LINE" \
						  --output-file-grp OCR-D-CIS-OCR-2 \
						  --mets "$tmpws/mets.xml" \
						  --parameter <(cat <<EOF
{
	"textequiv_level": "word",
	"model": "$tmpdir/download/fraktur2-00062000.pyrnn.gz"
}
EOF
)
ocrd-cis-align --log-level DEBUG \
			   -I OCR-D-CIS-OCR-1,OCR-D-CIS-OCR-2,OCR-D-GT-SEG-LINE \
			   -O OCR-D-CIS-ALIGN \
			   -m $tmpws/mets.xml

# fix ocr for some entries (otherwise the training will fail)
pushd $tmpws
for f in $(ocrd workspace find -G OCR-D-CIS-ALIGN); do
	sed -i -e 's#<pc:Unicode>e.</pc:Unicode>#<pc:Unicode>Säugethiere.</pc:Unicode>#' $f
	sed -i -e 's#<pc:Unicode>E</pc:Unicode>#<pc:Unicode>Säugethieren</pc:Unicode>#' $f
done
popd
rm -rf tmpws
cp -r $tmpws tmpws
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
	| sed -e "s#\${ocrd-cis-profiler-exe}#$tmpdir/bin/profiler.bash#" \
	| sed -e "s#\${ocrd-cis-trigrams}#$(ocrd-cis-data -3gs)#" \
	| sed -e "s#\${ocrd-cis-train-dir}#$tmpdir/train#"
)

if [[ ! -f $tmpdir/train/model.zip ]]; then
	echo $tmpdir/train/model.zip not found
	exit 1
fi
