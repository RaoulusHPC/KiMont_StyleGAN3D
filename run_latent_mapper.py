import argparse
import math
import os
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from training import dataset, visualize
from optimization.optimizer import LatentOptimizer
from models.base_models import Comparator, LatentMapper
from models.stylegan import StyleGAN
from train_stylegan import ModelParameters

if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')

    tfrecords = ['data/projected_images.tfrecords']
    tf_dataset = dataset.get_projected_dataset(tfrecords)

    tf_dataset = tf_dataset.map(lambda o, g, l, w : (g, w))
    test_dataset = tf_dataset.take(1000).batch(1)
    
    # c = 0
    # for d in train_dataset:
    #     for i in d:
    #         print(i.shape)
    #     original_image, generated_image, label, w = d
    #     if tf.math.reduce_mean(tf.math.square(original_image - generated_image)) < 0.005:
    #         break
    
    # def binarize(image):
    #     return tf.where(image > 0., 1., -1.)

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
    generator = model.generator_ema

    comparator = Comparator()
    comparator_input = tf.zeros(shape=(1, 64, 64, 64, 1))
    comparator(comparator_input, comparator_input)
    comparator.load_weights('tf_ckpt_comparator/')

    mapper = LatentMapper(generator.latent_size, num_layers=4)
    mapper.build((None, generator.latent_size))
    mapper.load_weights('tf_ckpt_mapper/')

    c = 0
    for generated_image, w in test_dataset:
        optimized_image = generator.synthesize(w + mapper(w))
        changes = tf.where((optimized_image - generated_image) > 0, 1., -1.)
        visualize.screenshot_and_save([generated_image, optimized_image, changes], filepath=f'lm{c}.png', shape=(1, 3), window_size=(1536, 512))
        c += 1
        if c == 5:
            break