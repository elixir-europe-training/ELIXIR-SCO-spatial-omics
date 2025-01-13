docker run \
--rm \
--name jupyter_user \
--user root \
-e NB_UMASK=002 \
-e CHOWN_EXTRA=/home/jovyan/workdir \
-e JUPYTER_ENABLE_LAB=yes \
-e JUPYTER_TOKEN=test \
--mount type=bind,source="$PWD",target=/home/jovyan/workdir \
-p 8888:8888 \
sco_jupyter \
start-notebook.sh
