FROM golang AS xpytest
RUN git clone --depth=1 https://github.com/chainer/xpytest.git /xpytest
RUN cd /xpytest && \
    go build -o /usr/local/bin/xpytest ./cmd/xpytest

ARG base_image=nvidia/cuda:10.0-devel-ubuntu18.04
FROM ${base_image}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends wget git g++ make


ADD find-apt-package-version /usr/local/bin

ARG CUTENSOR_VERSION=""
RUN [ ! -z "${CUTENSOR_VERSION}" ] && \
    apt-get install -y "$(find-apt-package-version libcutensor-dev ${CUTENSOR_VERSION})"

ARG NCCL_VERSION=""
RUN [ ! -z "${NCCL_VERSION}" ] && \
    apt-get install -y "$(find-apt-package-version libnccl-dev ${CUTENSOR_VERSION})"

ARG CUDNN_VERSION=""
RUN [ ! -z "${CUDNN_VERSION}" ] && \
    apt-get install -y "$(find-apt-package-version libcutensor-dev ${CUTENSOR_VERSION})"

RUN python3.7 -m pip install \
    'cython>=0.28.0' \
    'pytest==4.1.1' 'pytest-xdist==1.26.1' mock setuptools \
    filelock 'numpy>=1.9.0' 'protobuf==3.6.1'

COPY --from=xpytest /usr/local/bin/xpytest /usr/local/bin/xpytest
