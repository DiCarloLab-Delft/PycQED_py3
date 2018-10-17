FROM registry.gitlab.com/dicarlolab/pycqed/base_conda:latest

ARG GITHUB_PULL_USER
ARG GITHUB_PULL_TOKEN
ARG OPENQL_REVISION
ARG QCODES_REVISION

RUN mkdir /src && \
    cd /src && \
    git clone "https://$GITHUB_PULL_USER:$GITHUB_PULL_TOKEN@github.com/QE-Lab/OpenQL" openql && \
    cd openql && \
    git checkout $OPENQL_REVISION && \
    python setup.py build && \
    pip install --no-cache-dir . && \
    cd / && \
    rm -rf /src/openql && \
    pip install --no-cache-dir git+https://github.com/QCoDeS/Qcodes.git@$QCODES_REVISION

