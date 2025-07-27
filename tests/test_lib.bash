#/bin/bash

tmpdir=$(mktemp -d)
function stopserver() {
    :
}
function failexit() {
    stopserver
}
function cleanexit() {
    stopserver
    rm -rf $tmpdir
}
trap "trap failexit EXIT" ERR
trap cleanexit EXIT

OCRD_LOG_ARGS=()
if test -v OCRD_OVERRIDE_LOGLEVEL; then
    OCRD_LOG_ARGS+=(-l $OCRD_OVERRIDE_LOGLEVEL)
fi
OCRD_WS_ARGS=() # -m mets.xml

OCRD_CIS_FILEGRP="OCR-D-GT-SEG-LINE"

data_url="https://github.com/OCR-D/gt_structure_text/releases/download/v1.5.0/"
function ocrd_cis_download_bagit() {
	local url="$data_url/$1"
	mkdir -p "$PWD/download"
	wget -nc -P "$PWD/download" "$url"
}

function ocrd_cis_init_ws() {
	ocrd_cis_download_bagit "$1"
	ocrd zip spill -d "$tmpdir" "$PWD/download/$1"
	tmpws="$tmpdir/${1%.ocrd.zip}"
        if ((${OCRD_MAX_PARALLEL_PAGES:-0} > 1)); then
            echo starting METS server at $tmpws
            ocrd workspace -d "$tmpws" -U "$tmpws/mets.sock" server start &
            OCRD_WS_ARGS+=(-U "$tmpws/mets.sock")
            sleep 1
            function stopserver() {
                echo stopping METS server at $tmpws
                ocrd workspace -d "$tmpws" -U "$tmpws/mets.sock" server stop || true
            }
        fi
}


function ocrd_cis_align() {
	# download ocr models
	ocrd resmgr download ocrd-cis-ocropy-recognize fraktur.pyrnn.gz
	ocrd resmgr download ocrd-cis-ocropy-recognize fraktur-jze.pyrnn.gz
	# run ocr
        pushd $tmpws
        ARGS=(${OCRD_LOG_ARGS[*]} ${OCRD_WS_ARGS[*]})
        ocrd-cis-ocropy-binarize ${ARGS[*]} -I $OCRD_CIS_FILEGRP -O OCR-D-CIS-IMG-BIN
	ocrd-cis-ocropy-recognize ${ARGS[*]} -I OCR-D-CIS-IMG-BIN -O OCR-D-CIS-OCR-1 \
				-P textequiv_level word -P model fraktur.pyrnn.gz
	ocrd-cis-ocropy-recognize ${ARGS[*]} -I OCR-D-CIS-IMG-BIN -O OCR-D-CIS-OCR-2 \
				-P textequiv_level word -P model fraktur-jze.pyrnn.gz
	ocrd-cis-align ${ARGS[*]} -I OCR-D-CIS-OCR-1,OCR-D-CIS-OCR-2,$OCRD_CIS_FILEGRP \
				-O OCR-D-CIS-ALIGN
        popd
}

function fail() {
    echo >&2 "$@"
    false
}
