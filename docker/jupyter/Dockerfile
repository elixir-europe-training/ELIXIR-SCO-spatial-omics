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

# cellpose environment
ARG env_name=cellpose_env
RUN mamba env create -f /tmp/${env_name}.yml && \
    mamba clean --all -f -y
RUN "${CONDA_DIR}/envs/${env_name}/bin/python" -m ipykernel install --user --name="${env_name}" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# segmentation environment
ARG env_name=segmentation_env
RUN mamba env create -f /tmp/${env_name}.yml && \
    mamba clean --all -f -y
RUN "${CONDA_DIR}/envs/${env_name}/bin/python" -m ipykernel install --user --name="${env_name}" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"


# imaging based data analysis environment
ARG env_name=imaging_based_data_analysis_env
RUN mamba env create -f /tmp/${env_name}.yml && \
    mamba clean --all -f -y
RUN "${CONDA_DIR}/envs/${env_name}/bin/python" -m ipykernel install --user --name="${env_name}" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
