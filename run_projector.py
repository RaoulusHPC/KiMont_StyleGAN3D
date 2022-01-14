
import argparse
import numpy as np
import tensorflow as tf

from models.stylegan import StyleGAN
from training import dataset, visualize
from projection.projector import LatentProjector
from train import ModelParameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/example.npy", help="path to image")
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--screenshot", type=bool, default=False, help="whether to save a screenshot of the result")
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    # Load stylegan 

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

    c = 0
    b = 0
    for data, label in train_dataset:
        print('extended W+')
        projector = LatentProjector(model.generator_ema, steps=500, noise=0.01, screenshot=False)
        projector.init_w(label, w_avg_samples=10_000)
        loss1, generated_image = projector.project(data)
        print('W')
        projector = LatentProjector(model.generator_ema, space='W', steps=500, noise=0.01, screenshot=False)
        projector.init_w(label, w_avg_samples=10_000)
        loss2, generated_image = projector.project(data)

        if loss1 < loss2:
            b += 1
        c += 1
        if c > 100:
            break
    print(c, b)
