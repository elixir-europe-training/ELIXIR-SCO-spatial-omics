
FROM bioconductor/bioconductor_docker:RELEASE_3_20

USER root

COPY install.R /tmp/
RUN R -f /tmp/install.R && \
    R --version

