FROM ocrd/core:v2.62.0 AS base

WORKDIR /build-ocrd
COPY setup.py .
COPY ocrd_cis/ocrd-tool.json .
COPY ocrd_cis ./ocrd_cis
COPY README.md .
COPY Makefile .
RUN make install \
	&& rm -rf /build-ocrd

WORKDIR /data
