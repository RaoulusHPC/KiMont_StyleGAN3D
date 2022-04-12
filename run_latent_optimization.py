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
    train_dataset = train_dataset.skip(6100)
    # train_dataset = train_dataset.skip(5021)
    # train_dataset = train_dataset.skip(5052)

    model = StyleGAN(model_parameters=ModelParameters())
    model.build()
    model.setup_moving_average()
    ckpt = tf.train.Checkpoint(
        seen_images=tf.Variable(0),
        generator_ema=model.generator_ema)
    manager = tf.train.CheckpointManager(
        ckpt,
        directory='./ckpts/stylegan/',
        max_to_keep=None)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    model.trainable = False

    # visualize.visualize_tensors([original_image, generated_image, binarize(generated_image)], shape=(1, 3))
    comparator = Comparator()
    comparator_input = tf.zeros(shape=(1, 64, 64, 64, 1))
    comparator((comparator_input, comparator_input))
    comparator.load_weights('./ckpts/comparator/')
    comparator.trainable = False

    # temp_dataset = dataset.get_simplegrab_dataset('data/simpleGRAB_1000.tfrecords')
    # temp_dataset = temp_dataset.batch(1)
    # for components, label in temp_dataset:
    #     print(comparator(components[0], components[1], training=False), comparator(components[1], components[0], training=False)) 
    from optimization.optimizer import protected
    I = tf.keras.layers.ZeroPadding3D()(tf.ones((1, 62, 62, 62, 1)))
    S = I[:, :, :, :32]
    S = tf.concat([S, tf.zeros((1, 64, 64, 32, 1))], axis=3)
    O = I[:, :, :32]
    O = tf.concat([O, tf.zeros((1, 64, 32, 64, 1))], axis=2)
    # S = tf.concat([tf.zeros((1, 64, 64, 12, 1)), S], axis=3)
    visualize.screenshot_and_save([protected(O, S), S, O, I], filepath='protect.png', shape=(1, 4), window_size=(4000, 1000))
    #continue
    c = 0
    for d in train_dataset:
        c+=1
    print(c)
    for d in train_dataset:
        original_image, generated_image, label, w = d
        
        from optimization.optimizer import protected
        S = original_image[:, :, :, :48, :]
        S = tf.concat([S, tf.zeros((1, 64, 64, 16, 1))], axis=3)
        # S = tf.concat([tf.zeros((1, 64, 64, 12, 1)), S], axis=3)
        #visualize.visualize_tensors([protected(tf.ones_like(original_image), S), S, original_image], shape=(1, 3), window_size=(3000, 1000))
        #continue
        loss = 1.
        while loss > 0.1:
            optimizer = LatentOptimizer(model.generator_ema, comparator)
            optimized_image, w_opt, loss = optimizer.optimize(w)
            # optimizer = LatentOptimizer(model.generator_ema, comparator)
            # optimized_image, w_opt, loss = optimizer.optimize(w_opt)
        loss = 1.
        while loss > 0.1:
            optimizer = LatentOptimizer(model.generator_ema, comparator, protection='hard')
            optimized_image_protected, w_opt, loss = optimizer.optimize(w, S)
        visualize.visualize_tensors([optimized_image_protected, optimized_image, S, original_image], shape=(1, 4), window_size=(4000, 1000))
        visualize.screenshot_and_save([optimized_image_protected, optimized_image, S, original_image], filepath='results/optimization/protec_hard2.png', shape=(1, 4), window_size=(4000, 1000))


        # optimizer = LatentOptimizer(model.generator_ema, comparator, steps=200, grab_lambdas=(0., 1.), filepath=f'{c}test01.png')
        # optimized_image, w_opt = optimizer.optimize(w)

        # optimizer = LatentOptimizer(model.generator_ema, comparator, steps=200, grab_lambdas=(0.5, 0.5), filepath=f'{c}test11.png')
        # optimized_image, w_opt = optimizer.optimize(w)

        c += 1
        if c == 5:
            break