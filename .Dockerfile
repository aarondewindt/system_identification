FROM jupyter/scipy-notebook:latest

# Install system dependencies
USER root
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    vim \
    less \
    git

USER $NB_UID
COPY ./requirements.txt requirements.txt

# Install project as developer
RUN pip install -r requirements.txt
