FROM rocker/rstudio:4.4.2

RUN apt-get update && \
  apt-get install -y \
  imagemagick \
  curl \
  libpng-dev \
  libcurl4-openssl-dev \
  libxml2-dev \
  libssl-dev \
  pandoc \
  libfontconfig1-dev \
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
  libgmp-dev

RUN R -e "install.packages('BiocManager')"

COPY envs/install.R /tmp/

RUN R -f /tmp/install.R && \
    R --version

