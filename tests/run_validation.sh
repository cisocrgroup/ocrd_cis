#!/bin/bash

if ocrd ocrd-tool ocrd-tool.json validate | grep '<error>'; then
	exit 1
fi
