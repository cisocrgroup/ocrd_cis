ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE AS base
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://github.com/cisocrgroup/ocrd_cis/issues" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/cisocrgroup/ocrd_cis" \
    org.label-schema.build-date=$BUILD_DATE \
    org.opencontainers.image.vendor="DFG-Funded Initiative for Optical Character Recognition Development" \
    org.opencontainers.image.title="ocrd_cis" \
    org.opencontainers.image.description="Ocropy OCR and CIS post-correction bindings" \
    org.opencontainers.image.source="https://github.com/cisocrgroup/ocrd_cis" \
    org.opencontainers.image.documentation="https://github.com/cisocrgroup/ocrd_cis/blob/${VCS_REF}/README.md" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.base.name=ocrd/core

ENV GITURL="https://github.com/cisocrgroup"

SHELL ["/bin/bash", "-c"]

# deps
RUN apt-get update \
	&& apt-get -y install --no-install-recommends locales

# locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8

# install the profiler
FROM base AS profiler
RUN apt-get update \
	&& apt-get -y install --no-install-recommends cmake g++ libcppunit-dev libxerces-c-dev \
	&& git clone ${GITURL}/Profiler --branch devel --single-branch /build/Profiler \
	&& pushd /build/Profiler \
	&& cmake -DCMAKE_BUILD_TYPE=release . \
	&& make compileFBDic trainFrequencyList runDictSearch profiler \
	&& mkdir /apps \
	&& cp bin/compileFBDic bin/trainFrequencyList bin/profiler bin/runDictSearch /apps/ \
	&& popd \
    && rm -rf /build/Profiler

FROM profiler AS languagemodel
# install the profiler's language backend
COPY --from=profiler /apps/compileFBDic /apps/
COPY --from=profiler /apps/trainFrequencyList /apps/
COPY --from=profiler /apps/runDictSearch /apps/
RUN apt-get update \
	&& apt-get -y install --no-install-recommends icu-devtools \
	&& git clone ${GITURL}/Resources --branch master --single-branch /build/Resources \
	&& pushd /build/Resources/lexica \
	&& PATH=$PATH:/apps make \
	&& PATH=$PATH:/apps make test \
	&& PATH=$PATH:/apps make install \
	&& popd \
	&& rm -rf /build/Resources

FROM base AS postcorrection
# install ocrd_cis (python)
WORKDIR /build/ocrd_cis
COPY --from=languagemodel /etc/profiler/languages /etc/profiler/languages
COPY --from=profiler /apps/profiler /apps/
COPY --from=profiler /usr/lib/x86_64-linux-gnu/libicuuc.so /usr/lib//x86_64-linux-gnu/
COPY --from=profiler /usr/lib/x86_64-linux-gnu/libicudata.so /usr/lib//x86_64-linux-gnu/
COPY --from=profiler /usr/lib//x86_64-linux-gnu/libxerces-c-3.2.so /usr/lib//x86_64-linux-gnu/
COPY . .
COPY ocrd-tool.json .
# prepackage ocrd-tool.json as ocrd-all-tool.json
RUN ocrd ocrd-tool ocrd-tool.json dump-tools > $(dirname $(ocrd bashlib filename))/ocrd-all-tool.json
# prepackage ocrd-all-module-dir.json
RUN ocrd ocrd-tool ocrd-tool.json dump-module-dirs > $(dirname $(ocrd bashlib filename))/ocrd-all-module-dir.json
# install everything and reduce image size
RUN apt-get update \
	&& apt-get -y install --no-install-recommends gcc wget default-jre-headless \
	&& make install \
	# test always fail, resources not available for download. Resources should be made available
	# somewhere else, e.g. github.com/OCR-D/assets
	# && make test \
	&& rm -rf /build/ocrd_cis

WORKDIR /data
VOLUME /data
