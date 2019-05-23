FROM registry.gitlab.com/dicarlolab/pycqed/base:latest

ARG OPENQL_REVISION
ARG QCODES_REVISION

RUN pip install --no-cache-dir \
        git+https://github.com/QE-Lab/OpenQL.git@$OPENQL_REVISION \
        git+https://github.com/QCoDeS/Qcodes.git@$QCODES_REVISION
