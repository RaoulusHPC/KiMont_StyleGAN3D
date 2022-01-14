
FROM tensorflow/tensorflow:2.6.1-gpu

ARG USER_ID
ARG GROUP_ID

RUN apt-get update
RUN apt-get install libgl1-mesa-glx xvfb ffmpeg libsm6 libxext6  -y

RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install pyvista==0.31 scipy tqdm

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /stylegan3d

ENV AM_I_IN_A_DOCKER_CONTAINER Yes
