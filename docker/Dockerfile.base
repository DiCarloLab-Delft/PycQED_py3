FROM python:3.6-stretch
MAINTAINER Viacheslav Ostroukh <V.Ostroukh@tudelft.nl>

# make our environment sane
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    apt-get install --yes \
        cmake \
        gcc \
        g++ \
        libxkbcommon-x11-0 \
        make \
        swig \
        xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir Cython \
        # required by qcodes-0.1.11
        "matplotlib<2.3" \
        # required by qcodes-0.1.11
        "numpy<1.14" \
        "plotly<3.8" \
        PyQt5 \
        scipy \
        pandas && \
    pip install --no-cache-dir \
        adaptive \
        autodepgraph \
        cma \
        coverage \
        h5py \
        hsluv \
        httplib2 \
        ipython \
        ipywidgets \
        jsonschema \
        lmfit \
        networkx \
        spirack \
        pyzmq \
        "pygsti<0.9.6" \
        pyqtgraph \
        pyserial \
        pytest \
        pyvisa \
        qutip \
        qtpy \
        six \
        sip \
        sklearn \
        spirack \
        websockets \
        zhinst

RUN pip install --no-cache-dir \
        coverage \
        pytest-cov \
        codecov \
        codacy-coverage

ADD docker/xvfb_init /etc/init.d/xvfb
ADD docker/xvfb-daemon-run /usr/bin/xvfb-daemon-run
RUN chmod a+x /etc/init.d/xvfb && \
    chmod a+x /usr/bin/xvfb-daemon-run
