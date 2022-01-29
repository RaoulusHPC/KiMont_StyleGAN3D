# StyleGAN3D

https://dille.name/blog/2018/07/16/handling-file-permissions-when-writing-to-volumes-from-docker-containers/

docker build -t stylegan3d \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .

docker run --gpus all -it -v "$(pwd):/stylegan3d" --ipc=host stylegan3d

