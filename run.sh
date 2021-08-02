docker pull jupyter/datascience-notebook

docker run -d --rm --name jupyter_notebook \
    -p 8888:8888 \
    -v "${PWD}":/home/jovyan \
    jupyter/datascience-notebook

docker logs -f jupyter_notebook
