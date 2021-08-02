docker pull jupyter/datascience-notebook

docker run -d --name jupyter_notebook \
    -p 8888:8888 \
    -v "${PWD}":/home/jovyan/work \
    jupyter/datascience-notebook
