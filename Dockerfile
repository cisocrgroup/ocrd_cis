FROM flobar/profiler:debian AS profiler
ENV VERSION="Tue Feb 15 12:48:42 PM CET 2022"

FROM ocrd/core:latest AS base
FROM base AS postcorrection

# Install ocrd_cis
COPY --from=profiler /language-data /etc/profiler/languages
COPY --from=profiler /apps/profiler /apps/
COPY --from=profiler /usr/lib/x86_64-linux-gnu/libicuuc.so /usr/lib/x86_64-linux-gnu/
COPY --from=profiler /usr/lib/x86_64-linux-gnu/libicudata.so /usr/lib/x86_64-linux-gnu/
COPY --from=profiler /usr/lib/x86_64-linux-gnu/libxerces-c-3.2.so /usr/lib/x86_64-linux-gnu/
COPY --from=profiler /usr/lib/x86_64-linux-gnu/libcppunit-1.14.so /usr/lib/x86_64-linux-gnu/
COPY . /build
RUN apt-get update \
    && apt-get -y install --no-install-recommends gcc wget default-jre-headless \
    && cd /build \
    && make install \
    && make test \
    && cd / \
    && rm -rf /build
VOLUME ["/data"]
