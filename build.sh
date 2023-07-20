set -x
SUDO_STRING=`groups|grep docker`
SUDO=""
if [ -z "$SUDO_STRING" ]; then
  SUDO="sudo "
fi

DOCKER_BUILDKIT=1 $SUDO docker build \
    -t docker_gauss_newton_detector .