ARG TORCH_VERSION=2.0.1-cuda11.7-cudnn8
# ARG DEVEL=devel
ARG DEVEL=runtime

FROM pytorch/pytorch:${TORCH_VERSION}-${DEVEL} as build
COPY requirements*.txt /workspace/
RUN --mount=type=cache,target=/root/.cache/pip pip wheel --no-deps --wheel-dir /wheels -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip wheel --no-deps --wheel-dir /wheels-devel -r requirements-devel.txt


FROM pytorch/pytorch:${TORCH_VERSION}-runtime as base
RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get update && apt-get autoclean && DEBIAN_FRONTEND=noninteractive TZ=CEST \
    apt-get --yes install git libglib2.0-0 curl
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels --mount=type=cache,target=/root/.cache/pip pip install /wheels/*

RUN curl https://github.com/huggingface/pytorch-image-models/commit/94ca140b67cb602ee7e146af32bfb63b60df96f4.diff | patch -d /opt/conda/lib/python3.10/site-packages/ -p1


FROM base as devel
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get update && apt-get autoclean && DEBIAN_FRONTEND=noninteractive TZ=CEST \
    apt-get --yes install libx11-6 libgl1 libglu1 fonts-freefont-ttf sudo fontconfig
RUN --mount=type=bind,from=build,source=/wheels-devel,target=/wheels --mount=type=cache,target=/root/.cache/pip pip install /wheels/*

ARG USER=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $USER
RUN useradd -m -u $UID -g $GID -s /bin/bash -d /workspace/home $USER
RUN echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER && chmod 0440 /etc/sudoers.d/$USER
USER $USER

