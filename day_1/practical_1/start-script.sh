#!/bin/bash
# Activate the conda environment
source /opt/conda/bin/activate imaging_based_data_analysis_env
# Start the Jupyter Notebook server
exec jupyter notebook --allow-root --notebook-dir=/home/jovyan/workdir
