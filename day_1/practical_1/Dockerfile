# Use jupyter/minimal-notebook as the base image
FROM jupyter/minimal-notebook:latest
LABEL org.opencontainers.image.source="https://github.com/scilifelabdatacentre/serve-images"

ARG SERVE_BRANCH=develop

########################
# Switch to root to install system packages
########################
USER root

# Create /data directory so we can volume-mount it at runtime (shared, read-only)
RUN mkdir -p /home/jovyan/workdir /data /group_work

# Set working directory
WORKDIR /home/jovyan/workdir

########################
# Copy environment YAML files
########################
COPY envs/*.yml /tmp/envs/

########################
# Copy other relevant files into the image
########################
COPY requirements.txt /tmp/requirements.txt
COPY start-script.sh /usr/local/bin/start-script.sh
COPY workdir/ /home/jovyan/workdir/

########################
# Install system packages
########################
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl wget unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

########################
# Install Python packages in the base environment
########################
RUN pip install --no-cache-dir -r /tmp/requirements.txt

########################
# Install mamba in base environment
########################
RUN conda install -n base -c conda-forge mamba && conda clean --all --yes

########################
# Create ALL environments & register kernels
########################
RUN for f in /tmp/envs/*.yml; do \
      echo "Creating environment from $f" && \
      mamba env create -f "$f" && \
      conda clean --all --yes; \
    done && \
    # For each environment, install an IPython kernel so Jupyter can use it
    for env_name in $(ls /opt/conda/envs); do \
      if [ -f "/opt/conda/envs/$env_name/bin/python" ]; then \
        echo "Installing kernel for $env_name" && \
        /opt/conda/envs/"$env_name"/bin/python -m ipykernel install --user \
            --name "$env_name" --display-name "Python ($env_name)"; \
      fi; \
    done

########################
# Fix Ownership & Permissions
########################
# Set ownership and permissions according to each directory’s needs
RUN chown -R jovyan:users /home/jovyan /opt/conda /home/jovyan/workdir /data /group_work && \
    \
    # 1) /home/jovyan and /home/jovyan/workdir: read-write for all
    chmod -R a+rwx /home/jovyan /home/jovyan/workdir /opt/conda && \
    \
    # 2) /data: read-only for all (a+rx)
    chmod -R a+rx /data && \
    \
    # 3) /group_work: read-write for all (a+rwx)
    chmod -R a+rwx /group_work && \
    \
    # Make sure the startup script is executable
    chmod +x /usr/local/bin/start-script.sh

########################
# Switch back to jovyan user
########################
USER jovyan

########################
# Set default conda environment
########################
ENV DEFAULT_ENV=imaging_based_data_analysis_env
ENV PATH=/opt/conda/envs/${DEFAULT_ENV}/bin:$PATH

########################
# Disable Jupyter token authentication
########################
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /home/jovyan/.jupyter/jupyter_notebook_config.py

########################
# Set the startup script as the container entrypoint
########################
ENTRYPOINT ["/usr/local/bin/start-script.sh"]
