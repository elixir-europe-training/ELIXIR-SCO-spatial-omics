FROM jupyter/minimal-notebook:latest

# change user to root to modify .bashrc for e.g. highlighting in the console
USER root
RUN cat /etc/skel/.bashrc >> /etc/bash.bashrc

# change back to notebook user
USER $NB_UID

# remove auto-generated work directory
RUN rm -r /home/jovyan/work

# intall required software with conda
COPY environments/*.yml /tmp

# install environments
# For each environment, install an IPython kernel so Jupyter can use it
RUN for f in /tmp/envs/*.yml; do \
      echo "Creating environment from $f" && \
      mamba env create -f "$f" && \
      conda clean --all --yes; \
    done && \
    for env_name in $(ls /opt/conda/envs); do \
      if [ -f "/opt/conda/envs/$env_name/bin/python" ]; then \
        echo "Installing kernel for $env_name" && \
        "${CONDA_DIR}/envs/${env_name}/bin/python" -m ipykernel install --user --name="${env_name}" && \
        fix-permissions "${CONDA_DIR}" && \
        fix-permissions "/home/${NB_USER}"; \
      fi; \
    done
