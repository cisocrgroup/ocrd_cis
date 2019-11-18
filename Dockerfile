FROM ocrd/core:latest
ENV VERSION="Mi 9. Okt 13:26:16 CEST 2019"
ENV GITURL="https://github.com/cisocrgroup"
ENV DOWNLOAD_URL="http://cis.lmu.de/~finkf"
ENV DATA="/apps/ocrd-cis-post-correction"

# deps
COPY data/docker/deps.txt ${DATA}/deps.txt
RUN apt-get update \
	&& apt-get -y install --no-install-recommends $(cat ${DATA}/deps.txt)

# locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8

# install the profiler
RUN	git clone ${GITURL}/Profiler --branch devel --single-branch /tmp/profiler \
	&& cd /tmp/profiler \
	&& mkdir build \
	&& cd build \
	&& cmake -DCMAKE_BUILD_TYPE=release .. \
	&& make compileFBDic trainFrequencyList profiler \
	&& cp bin/compileFBDic bin/trainFrequencyList bin/profiler /apps/ \
	&& cd / \
    && rm -rf /tmp/profiler

# install the profiler's language backend
RUN	git clone ${GITURL}/Resources --branch master --single-branch /tmp/resources \
	&& cd /tmp/resources/lexica \
	&& make FBDIC=/apps/compileFBDic TRAIN=/apps/trainFrequencyList \
	&& mkdir -p /${DATA}/languages \
	&& cp -r german latin greek german.ini latin.ini greek.ini /${DATA}/languages \
	&& cd / \
	&& rm -rf /tmp/resources

# install ocrd_cis (python)
COPY Manifest.in Makefile setup.py ocrd-tool.json /tmp/build/
COPY ocrd_cis/ /tmp/build/ocrd_cis/
COPY bashlib/ /tmp/build/bashlib/
# COPY . /tmp/ocrd_cis
RUN cd /tmp/build \
	&& make install \
	&& cd / \
	&& rm -rf /tmp/build

# download ocr models and pre-trainded post-correction model
RUN mkdir /apps/models \
	&& cd /apps/models \
	&& wget ${DOWNLOAD_URL}/model.zip >/dev/null 2>&1 \
	&& wget ${DOWNLOAD_URL}/fraktur1-00085000.pyrnn.gz >/dev/null 2>&1 \
	&& wget ${DOWNLOAD_URL}/fraktur2-00062000.pyrnn.gz >/dev/null 2>&1

VOLUME ["/data"]
