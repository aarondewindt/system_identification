FROM jupyter/scipy-notebook:lab-3.1.13

## Define environment variables
ENV PROJECT_NAME=system_identification
ENV PROJECT_DIR=$HOME/$PROJECT_NAME

# Switch to the root user
USER root

# Copy configuration files
COPY config_files/overrides.json /opt/conda/share/jupyter/lab/settings/overrides.json
COPY config_files/pycodestyle $HOME/.config/pycodestyle
COPY config_files/mypy_config $HOME/.config/mypy/config

# Copy project
COPY ./ $PROJECT_DIR

# Install system dependencies
RUN sudo apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    vim \
    less \
  && rm -rf /var/lib/apt/lists/*

# Install Jupterlab and extentions
RUN pip install \
    jupyterlab-lsp \
    'python-language-server[all]' \
    pyls-mypy \
    lckr-jupyterlab-variableinspector \
    ipywidgets \
    aquirdturtle_collapsible_headings \
    jupyterlab-spellchecker \
    ipympl \
    jupyterlab_widgets

# Change ownership to the user
RUN chown -R $NB_UID:$NB_GID $PROJECT_DIR
RUN chown -R $NB_UID:$NB_GID $HOME/.config

# Switch to jupyter user
USER $NB_UID
WORKDIR $HOME

# Install project as developer
RUN pip install -e $PROJECT_DIR --user
