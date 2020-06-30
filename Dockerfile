FROM ocrd/core:latest AS base
ENV VERSION="Di 12. Mai 13:26:35 CEST 2020"
ENV GITURL="https://github.com/cisocrgroup"
ENV DOWNLOAD_URL="http://cis.lmu.de/~finkf"

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
	&& git clone ${GITURL}/Profiler --branch devel --single-branch /build \
	&& cd /build \
	&& cmake -DCMAKE_BUILD_TYPE=release . \
	&& make compileFBDic trainFrequencyList runDictSearch profiler \
	&& mkdir /apps \
	&& cp bin/compileFBDic bin/trainFrequencyList bin/profiler bin/runDictSearch /apps/ \
	&& cd / \
    && rm -rf /build

FROM profiler AS languagemodel
# install the profiler's language backend
COPY --from=profiler /apps/compileFBDic /apps/
COPY --from=profiler /apps/trainFrequencyList /apps/
COPY --from=profiler /apps/runDictSearch /apps/
RUN apt-get update \
	&& apt-get -y install --no-install-recommends icu-devtools \
	&& git clone ${GITURL}/Resources --branch master --single-branch /build \
	&& cd /build/lexica \
	&& PATH=$PATH:/apps make \
	&& PATH=$PATH:/apps make test \
	&& PATH=$PATH:/apps make install \
	&& cd / \
	&& rm -rf /build

FROM base AS postcorrection
# install ocrd_cis (python)
VOLUME ["/data"]
COPY --from=languagemodel /etc/profiler/languages /etc/profiler/languages
COPY --from=profiler /apps/profiler /apps/
COPY --from=profiler /usr/lib/x86_64-linux-gnu/libicuuc.so /usr/lib//x86_64-linux-gnu/
COPY --from=profiler /usr/lib/x86_64-linux-gnu/libicudata.so /usr/lib//x86_64-linux-gnu/
COPY --from=profiler /usr/lib//x86_64-linux-gnu/libxerces-c-3.2.so /usr/lib//x86_64-linux-gnu/
COPY . /build
RUN apt-get update \
	&& apt-get -y install --no-install-recommends gcc wget default-jre-headless \
	&& cd /build \
	&& make install \
	&& make test \
	&& cd / \
	&& rm -rf /build
