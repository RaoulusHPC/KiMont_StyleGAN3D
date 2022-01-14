import argparse
import math
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from training import dataset, visualize
from optimization.optimizer import LatentOptimizer
from models.base_models import Comparator
from models.stylegan import StyleGAN
from train import ModelParameters

if __name__ == "__main__":

    train_dataset = dataset.get_projected_dataset('data/projected_images.tfrecords')
    train_dataset = train_dataset.skip(5)
    c = 0
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

    # visualize.visualize_tensors([original_image, generated_image, binarize(generated_image)], shape=(1, 3))
    comparator = Comparator()
    comparator.build((None, 64, 64, 64, 2))
    comparator.load_weights('tf_ckpt_comparator/')
    print(comparator.predict(tf.concat([original_image, generated_image], axis=-1)))

    optimizer = LatentOptimizer(model.generator_ema, comparator, filepath='test1.png')
    optimized_image, w_opt = optimizer.optimize(w)