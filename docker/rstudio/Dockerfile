
FROM bioconductor/bioconductor_docker:RELEASE_3_20

USER root

RUN apt-get update && \
  apt-get install -y \
  curl \
  libpng-dev \
  libcurl4-openssl-dev \
  libxml2-dev \
  libssl-dev \
  pandoc \
  libfontconfig1-dev \
  nano \
  libhdf5-dev \
  default-jdk \
  cmake \
  libfftw3-dev \
  libgeos-dev \
  libmagick++-dev \
  libproj-dev \
  libgdal-dev \
  libharfbuzz-dev \
  libfribidi-dev  \
  libudunits2-dev \
  libgsl-dev \
  libgmp3-dev \
  libglpk40 \
  patch \
  libglpk-dev \
  libgmp-dev \
  libtiff5-dev \
  libtiff-dev

COPY envs/install.R /tmp/
RUN R -f /tmp/install.R && \
    R --version

