#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))

DOCKER_VOLUME="${DOCKER_VOLUME} -v $(dirname ${RUN_DIR}):/workspace/involution:rw"

docker run \
    -it \
    --rm \
    --gpus '"device=0"' \
    ${DOCKER_VOLUME} \
    --name Involution-PyTorch \
    pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel bash
    # pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel bash
    # pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel bash
    # pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel bash
    # nvcr.io/nvidia/pytorch:21.05-py3
    # nvcr.io/nvidia/pytorch:20.08-py3
