#!/bin/bash

iname=${DOCKER_IMAGE:-"docker_gauss_newton_detector"}
cname=${DOCKER_CONTAINER:-"gauss_newton_detector"} ## name of container (should be same as in exec.sh)

DEFAULT_USER_DIR=${USER_DIR:-"$(pwd)"}
mtdir=${MOUNTED_DIR:-$DEFAULT_USER_DIR}

VAR=${@:-"bash"}
if [ $# -eq 0 -a -z "$OPT" ]; then
    OPT=-it
fi

## --net=mynetworkname
## docker inspect -f '{{.NetworkSettings.Networks.mynetworkname.IPAddress}}' container_name
## docker inspect -f '{{.NetworkSettings.Networks.mynetworkname.Gateway}}'   container_name

DOCKER_ENVIRONMENT_VAR=""

XDG_OPTION="-u $(id -u):$(id -g) --volume=/run/user/$(id -u):/tmp/xdg_runtime --env=XDG_RUNTIME_DIR=/tmp/xdg_runtime"
##xhost +local:root
xhost +si:localuser:root

docker rm ${cname}

docker run \
    --privileged     \
    ${OPT}           \
    ${DOCKER_ENVIRONMENT_VAR} \
    ${XDG_OPTION} \
    --env="DISPLAY"  \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --name=${cname} \
    --volume="${mtdir}/src:/workspace/src" \
    -w="/workspace/src" \
    ${iname} \
    ${VAR}

##xhost -local:root
## capabilities
# compute	CUDA / OpenCL アプリケーション
# compat32	32 ビットアプリケーション
# graphics	OpenGL / Vulkan アプリケーション
# utility	nvidia-smi コマンドおよび NVML
# video		Video Codec SDK
# display	X11 ディスプレイに出力
# all