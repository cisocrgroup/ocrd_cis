FROM ocrd/core:edge
MAINTAINER OCR-D
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONIOENCODING utf8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /build
COPY setup.py .
COPY ocrd_cis ./ocrd_cis
RUN pip3 install --upgrade pip .

ENTRYPOINT ["/bin/sh", "-c"]
