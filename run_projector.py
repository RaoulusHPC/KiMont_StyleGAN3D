
import argparse
import os
import numpy as np
import tensorflow as tf

from models.stylegan import StyleGAN
from training import dataset, visualize
from projection.projector import LatentProjector
from train_stylegan import ModelParameters

if __name__ == "__main__":

    # Load stylegan 

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
    #image = np.load(args.file_path)
    tfrecords = ['data/mcb64_screws.tfrecords']
    train_dataset = dataset.get_mcb_dataset(
        tfrecords=tfrecords,
        batch_size=1,
        repeat=1,
        augment_function=None)

    # for data, label in train_dataset:
    #     print(data.shape, label.shape)
    #     # data = data[0]
    #     # label = label[0]
    #     # visualize.visualize_tensors([data])
    #     break

    save_dir = f'./results/inversion'
    os.makedirs(save_dir, exist_ok=True)

    c = 0
    b = 0

    for data, label in train_dataset:
        # print('extended W+')
        # projector = LatentProjector(model.generator_ema, steps=500, noise=0.01, screenshot=False)
        # projector.init_w(label, w_avg_samples=10_000)
        # loss1, generated_image = projector.project(data)
        print('W')
        projector = LatentProjector(model.generator_ema, space='W', steps=1000, noise=0.01, screenshot=False)
        projector.init_w(label, w_avg_samples=10_000)
        loss2, generated_image = projector.project(data)

        projector = LatentProjector(model.generator_ema, space='W', steps=500, noise=0.01, screenshot=False)
        projector.init_w(label, w_avg_samples=10_000)
        loss2, generated_image = projector.project(data)

        projector = LatentProjector(model.generator_ema, space='W', steps=250, noise=0.01, screenshot=False)
        projector.init_w(label, w_avg_samples=10_000)
        loss2, generated_image = projector.project(data)

        save_filepath = os.path.join(save_dir, str(c))
        os.makedirs(save_dir, exist_ok=True)
        visualize.screenshot_and_save([generated_image, data], filepath=save_filepath, shape=(1, 2), window_size=(2000, 1000))

        # if loss1 < loss2:
        #     b += 1
        c += 1
        if c > 100:
            break
    print(c, b)
