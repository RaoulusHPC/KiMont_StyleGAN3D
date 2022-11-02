
import argparse
import numpy as np
import tensorflow as tf
import random

from training import dataset, visualize
from projection.projector import LatentProjector
from train_stylegan import ModelParameters
from training.visualize import screenshot_and_save

class TrainingArguments:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.5
    train_size: int = 5000
    l2_lambda: float = 2.0


if __name__ == "__main__":

    parameters = TrainingArguments()


    from models.stylegan import StyleGAN

    model = StyleGAN(model_parameters=ModelParameters())
    model.build()
    model.setup_moving_average()

    ckpt = tf.train.Checkpoint(
        seen_images=tf.Variable(0),
        generator_ema=model.generator_ema)
    manager = tf.train.CheckpointManager(
        ckpt,
        directory='./tf_ckpts',
        max_to_keep=None)

    ckpt.restore('./tf_ckpts/ckpt-20').expect_partial() #manager.latest_checkpoint
    print("Loaded weights from ckpt-20")
    
    tfrecords = ['data/projected_images.tfrecords']
    tf_dataset = dataset.get_projected_dataset(tfrecords)

    tf_dataset = tf_dataset.map(lambda o, g, l, w : (g, w, l))
    train_dataset = tf_dataset.take(parameters.train_size).shuffle(parameters.train_size, reshuffle_each_iteration=True).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf_dataset.skip(parameters.train_size).shuffle(parameters.train_size, reshuffle_each_iteration=True).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)   
    
    for generated_image, w, l  in train_dataset:
    	print(generated_image.shape)
    	output = model.generator_ema.synthesize(w)
    	index = random.randint(1,32)
    	comp = []
    	comp.append(generated_image[index,:,:,:,:])
    	comp.append(output[index,:,:,:,:])
    	print(comp[0].shape)
    	screenshot_and_save(comp, filepath='logs/viz_latents/test.png')
    	break
    
