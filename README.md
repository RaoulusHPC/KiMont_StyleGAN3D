# StyleGAN3D

### Pipeline

![image info](./documentation/StyleGAN_Diagramm(1).png)





### Step-by-Step Instructions

1. Using the MCB dataset, train the StyleGAN with [train_stylegan.py](./train_stylegan.py)
2. As for now, the model overfits after approximately 20 epochs, therefore, for further training, use a
training checkpoint between epoch 20 and 25:
```
ckpt.restore('./tf_ckpts/ckpt-20').expect_partial() 
```
3. Using the trained StyleGAN and the MCB Dataset create the latent dataset from the MCB dataset with [create_latent_dataset.py](./create_latent_dataset.py) . For further
processing, we convert the objects from MCB to their latent representation. The resulting dataset can be found under
**data/latents.tfrecords**.
4. Next, we need to train the comparator. The comparator will guide all the future optimizations. The comparator decides whether
a CAD model is better or worse in some aspect than another model. In order to train the comparator, we need the [simpleGRAB](data/simpleGRAB_1000.tfrecords)
dataset.
5. We now have all the different sub-models, in order to train the latent mapper. The latent mapper will make small changes to the latent representation of a CAD model,
in order to improve it in some aspect. For the training, we need the latent dataset, the StyleGAN in order to go convert latent codes to CAD models and the 
comparator to guide the latent mapper.
6 To be continued..

### Docker installation
https://dille.name/blog/2018/07/16/handling-file-permissions-when-writing-to-volumes-from-docker-containers/

docker build -t stylegan3d \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .

docker run --gpus all -it -v "$(pwd):/stylegan3d" --ipc=host stylegan3d

