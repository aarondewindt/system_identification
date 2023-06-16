FROM jupyter/scipy-notebook:9e3ab9075a5e

# Install system dependencies
USER root

RUN apt-get update
RUN apt install -y -qq --no-install-recommends software-properties-common
RUN add-apt-repository universe
RUN apt-get update
RUN apt-get install -y -qq --no-install-recommends \
    vim \
    less \
    git \
    htop \
    texlive-full \ 
    texlive-bibtex-extra

USER $NB_UID
COPY ./requirements.txt requirements.txt

# Install project as developer
RUN pip install -r requirements.txt
