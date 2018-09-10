FROM continuumio/miniconda3:latest
MAINTAINER Viacheslav Ostroukh <V.Ostroukh@tudelft.nl>

# make our environment sane
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    apt-get install --yes \
        cmake \
        gcc \
        g++ \
        make \
        swig \
        xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda config --add channels conda-forge && \
    conda update --all --yes && \
    conda install --yes \
        adaptive \
        coverage \
        cython \
        h5py \
        ipython \
        ipywidgets \
        jsonschema \
        lmfit \
        # required by qcodes-0.1.11
        "matplotlib<2.3" \
        # required by qcodes-0.1.11
        "numpy<1.14" \
        plotly \
        pyqtgraph \
        pyserial \
        pyvisa \
        pyzmq \
        qutip \
        scipy \
        six \
        sip \
        pandas \
        pyqt \
        pytest \
        qtpy \
        websockets && \
    conda clean --all --yes && \
    pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
        autodepgraph \
        cma \
        hsluv \
        pygsti \
        spirack \
        zhinst

ADD docker/xvfb_init /etc/init.d/xvfb
ADD docker/xvfb-daemon-run /usr/bin/xvfb-daemon-run
RUN chmod a+x /etc/init.d/xvfb && \
    chmod a+x /usr/bin/xvfb-daemon-run
