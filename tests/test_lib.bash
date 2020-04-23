#/bin/bash
set -e

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
