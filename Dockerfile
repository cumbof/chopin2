# Title          :chopin2
# Description    :Supervised Classification with Hyperdimensional Computing
# Author         :Fabio Cumbo (fabio.cumbo@gmail.com)
# License        :https://github.com/fabio-cumbo/chopin2/blob/master/LICENSE

FROM ubuntu:18.04

MAINTAINER fabio.cumbo@gmail.com

# Set the working directory
WORKDIR /home

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Installing basic dependancies
RUN apt-get install -y \ 
        build-essential \
        curl \
        git \
        python3.8 \
        python3.8-dev \
        python3.8-venv \
        python3.8-distutils

# Make Python 3.8 available with venv
RUN python3.8 -m venv /venv
ENV PATH="/venv/bin:${PATH}"

# Upgrade pip using Python 3.8
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# Download chopin2
RUN mkdir -p /home/git && cd /home/git && \
    git clone https://github.com/fabio-cumbo/chopin2 && \
    cd /home/git/chopin2/src && \
    chmod +x chopin2.py functions.py
# Install Python dependencies
RUN pip install -r /home/git/chopin2/src/requirements.txt --ignore-installed
# Add chopin2 to PATH
ENV PATH="${PATH}:/home/git/chopin2/src"

WORKDIR /home
ENTRYPOINT /bin/bash