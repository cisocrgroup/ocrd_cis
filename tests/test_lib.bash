#/bin/bash

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

OCRD_CIS_FILEGRP="OCR-D-GT-SEG-LINE"
data_url="https://ocr-d-repo.scc.kit.edu/api/v1/dataresources/75ad9f94-dbaa-43e0-ab06-2ce24c497c61/data"
function ocrd_cis_download_bagit() {
	local url="$data_url/$1"
	mkdir -p "$tmpdir/download"
	wget -P "$tmpdir/download" "$url"
}

function ocrd_cis_init_ws() {
	ocrd_cis_download_bagit "$1"
	ocrd zip spill -d "$tmpdir" "$tmpdir/download/$1"
	tmpws="$tmpdir/${1%.ocrd.zip}"
}

function ocrd_cis_align() {
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
}
