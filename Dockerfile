# syntax=docker/dockerfile:1 
FROM ubuntu:22.04

ARG OPENCV_VERSION='4.8.0'

# https://qiita.com/haessal/items/0a83fe9fa1ac00ed5ee9
ENV DEBCONF_NOWARNINGS=yes
# https://qiita.com/yagince/items/deba267f789604643bab
ENV DEBIAN_FRONTEND=noninteractive
# https://qiita.com/jacob_327/items/e99ca1cf8167d4c1486d
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# https://stackoverflow.com/a/25423366
SHELL ["/bin/bash", "-c"]

# https://genzouw.com/entry/2019/09/04/085135/1718/
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

# Install basic packages
RUN apt-get update -qq && \
    apt-get install -y \
    aptitude build-essential \
    lsb-release \
    tzdata \
    wget \
    gnupg2 \
    curl \
    emacs \
    cmake \
    clang \
    libssl-dev \
    git \
    unzip \
    libgtk2.0-dev \
    pkg-config

RUN aptitude update -q

RUN mkdir -p workspace
WORKDIR /workspace

# Media I/O:
RUN apt install -y \
    zlib1g-dev \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    libtiff5-dev \
    libopenexr-dev \
    libgdal-dev \
    libgtk2.0-dev

# Parallelism and linear algebra libraries:
RUN apt install -y \
    libtbb-dev \
    libeigen3-dev

# Build OpenCV
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    cd OpenCV && \
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    mkdir build && \
    cd build && \
    cmake \
      -D WITH_TBB=ON \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_EXAMPLES=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_V4L=ON \
      -D WITH_OPENGL=ON \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=${GPU_ARCH} \
      -D CUDA_ARCH_PTX=${GPU_ARCH} \
      -D WITH_CUBLAS=ON \
      -D WITH_CUFFT=ON \
      -D WITH_EIGEN=ON \
      -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules/ \
      .. && \
    make all -j$(nproc) && \
    make install

# Install GMP MPFR MPC gcc-12
RUN apt-get update -qq && \
    apt-get install -y \
    m4 \
    libgmp-dev \
    libmpfr-dev \
    gcc-12

# Install GTK
RUN apt-get update -qq && \
    apt-get install -y \
    libcanberra-gtk-module \
    libcanberra-gtk3-module