from pathlib import Path

import tensorflow as tf
from training.visualize import visualize_tensors

from metrics.frechet_inception_distance import calculate_screenshot_fid, calculate_slice_fid
from training import dataset
from models.stylegan import StyleGAN
import train_stylegan

model_parameters = train_stylegan.ModelParameters()

dataset_name = 'ABC'
if dataset_name == 'MCB':
    tfrecords = ['data/mcb64_screws.tfrecords']
    tf_dataset = dataset.get_mcb_base(tfrecords)
    model_parameters.label_size = 9
elif dataset_name == 'ABC':
    tfrecords = list(Path('data/abc/').rglob('*.tfrecords'))
    tf_dataset = dataset.get_abc_base(tfrecords)
    model_parameters.label_size = 0

train_dataset = tf_dataset.take(50000).batch(64).prefetch(tf.data.AUTOTUNE)

    
model = StyleGAN(model_parameters=model_parameters)
model.build()
model.setup_moving_average()

# model.summary()

ckpt = tf.train.Checkpoint(
    seen_images=tf.Variable(0),
    generator_ema=model.generator_ema)
manager = tf.train.CheckpointManager(
    ckpt,
    directory='./ckpts/stylegan_abc',
    max_to_keep=None)

logger1 = tf.summary.create_file_writer('logs/metrics/abc/sag_fid')
logger2 = tf.summary.create_file_writer('logs/metrics/abc/axi_fid')
logger3 = tf.summary.create_file_writer('logs/metrics/abc/cor_fid')

for checkpoint in manager.checkpoints[::1]:
    ckpt.restore(checkpoint).expect_partial()
    seen_k_images = int(ckpt.seen_images // 1000)
    print(seen_k_images)
    # screenshot_fid = calculate_screenshot_fid(model.mapping_network_ema, model.generator_ema, 10_000, real_dataset, 64)
    sag_fid, axi_fid, cor_fid = calculate_slice_fid(model.generator_ema, 1, train_dataset, 64)
    with logger1.as_default():
        tf.summary.scalar(name='slice-wise-fids', data=sag_fid, step=seen_k_images)
    with logger2.as_default():
        tf.summary.scalar(name='slice-wise-fids', data=axi_fid, step=seen_k_images)
    with logger3.as_default():
        tf.summary.scalar(name='slice-wise-fids', data=cor_fid, step=seen_k_images)
