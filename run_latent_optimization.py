import argparse
import math
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from training import dataset, visualize
from optimization.optimizer import LatentOptimizer
from models.base_models import Comparator
from models.stylegan import StyleGAN
from train_stylegan import ModelParameters

if __name__ == "__main__":

    train_dataset = dataset.get_projected_dataset('data/projected_images.tfrecords')
    train_dataset = train_dataset.batch(1)
    train_dataset = train_dataset.skip(5)

    for d in train_dataset:
        original_image, generated_image, label, w = d
        if tf.math.reduce_mean(tf.math.square(original_image - generated_image)) < 0.005:
            break
    
    def binarize(image):
        return tf.where(image > 0., 1., -1.)

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
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    model.trainable = False

    # visualize.visualize_tensors([original_image, generated_image, binarize(generated_image)], shape=(1, 3))
    comparator = Comparator()
    comparator_input = tf.zeros(shape=(1, 64, 64, 64, 1))
    comparator(comparator_input, comparator_input)
    comparator.load_weights('tf_ckpt_comparator/')
    comparator.trainable = False

    # temp_dataset = dataset.get_simplegrab_dataset('data/simpleGRAB_1000.tfrecords')
    # temp_dataset = temp_dataset.batch(1)
    # for components, label in temp_dataset:
    #     print(comparator(components[0], components[1], training=False), comparator(components[1], components[0], training=False)) 

    c = 0
    for d in train_dataset:
        original_image, generated_image, label, w = d
            
        optimizer = LatentOptimizer(model.generator_ema, comparator, steps=200, grab_lambdas=(1., 0.), filepath=f'{c}test10.png')
        optimized_image, w_opt = optimizer.optimize(w)

        optimizer = LatentOptimizer(model.generator_ema, comparator, steps=200, grab_lambdas=(0., 1.), filepath=f'{c}test01.png')
        optimized_image, w_opt = optimizer.optimize(w)

        optimizer = LatentOptimizer(model.generator_ema, comparator, steps=200, grab_lambdas=(0.5, 0.5), filepath=f'{c}test11.png')
        optimized_image, w_opt = optimizer.optimize(w)

        c += 1
        if c == 5:
            break