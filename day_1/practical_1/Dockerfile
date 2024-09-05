FROM jupyter/minimal-notebook:latest
LABEL org.opencontainers.image.source=https://github.com/scilifelabdatacentre/serve-images

ARG SERVE_BRANCH=develop
WORKDIR /home/jovyan/workdir

COPY requirements.txt /tmp
COPY imaging_based_data_analysis_env.yml /tmp
COPY lab /home/jovyan/workdir/lab
COPY start-script.sh /usr/local/bin/start-script.sh

USER root
RUN apt-get update && apt-get install curl -y --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN conda install -n base -c conda-forge mamba

RUN mamba env create -f /tmp/imaging_based_data_analysis_env.yml && conda clean --all --yes

RUN echo "conda activate imaging_based_data_analysis_env" >> /etc/profile.d/conda.sh

RUN /opt/conda/envs/imaging_based_data_analysis_env/bin/pip install ipykernel && \
    /opt/conda/envs/imaging_based_data_analysis_env/bin/python -m ipykernel install --name imaging_based_data_analysis_env --display-name "Python (imaging_based_data_analysis_env)"

RUN chown -R jovyan:users /opt/conda /home/jovyan/.cache /home/jovyan/workdir && \
    chmod -R a+rwx /home/jovyan/workdir && \
    chmod +x /usr/local/bin/start-script.sh

USER jovyan

# Set default conda environment
ENV PATH=/opt/conda/envs/imaging_based_data_analysis_env/bin:$PATH

# Disable Jupyter token authentication
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /home/jovyan/.jupyter/jupyter_notebook_config.py

# Set the startup script as the entrypoint
ENTRYPOINT ["/usr/local/bin/start-script.sh"]
