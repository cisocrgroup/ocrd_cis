#/bin/bash

tmpdir=$(mktemp -d)
trap "trap 'echo exiting without removing $tmpdir' EXIT" ERR
trap "rm -rf $tmpdir" EXIT

OCRD_CIS_FILEGRP="OCR-D-GT-SEG-LINE"
data_url="https://github.com/OCR-D/gt_structure_text/releases/download/l1.1.19/"
function ocrd_cis_download_bagit() {
	local url="$data_url/$1"
	mkdir -p "$PWD/download"
	wget -nc -P "$PWD/download" "$url"
}

function ocrd_cis_init_ws() {
	ocrd_cis_download_bagit "$1"
	ocrd zip spill -d "$tmpdir" "$PWD/download/$1"
	tmpws="$tmpdir/${1%.ocrd.zip}"
}

function ocrd_cis_align() {
	# download ocr models
	ocrd resmgr download ocrd-cis-ocropy-recognize fraktur.pyrnn.gz
	ocrd resmgr download ocrd-cis-ocropy-recognize fraktur-jze.pyrnn.gz
	# run ocr
        pushd $tmpws
        ocrd-cis-ocropy-binarize -l DEBUG -I $OCRD_CIS_FILEGRP -O OCR-D-CIS-IMG-BIN
	ocrd-cis-ocropy-recognize -l DEBUG \
 				-I OCR-D-CIS-IMG-BIN -O OCR-D-CIS-OCR-1 \
				-P textequiv_level word -P model fraktur.pyrnn.gz
	ocrd-cis-ocropy-recognize -l DEBUG \
				-I OCR-D-CIS-IMG-BIN -O OCR-D-CIS-OCR-2 \
				-P textequiv_level word -P model fraktur-jze.pyrnn.gz
	ocrd-cis-align -l DEBUG	-I OCR-D-CIS-OCR-1,OCR-D-CIS-OCR-2,$OCRD_CIS_FILEGRP \
				-O OCR-D-CIS-ALIGN
        popd
}

function fail() {
    echo >&2 "$@"
    false
}
