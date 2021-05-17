#!/bin/bash
set -e

if ocrd ocrd-tool ocrd-tool.json validate | grep '<error>'; then
	exit 1
fi
