from pathlib import Path

import tensorflow as tf
from training.visualize import visualize_tensors

from metrics.frechet_inception_distance import calculate_screenshot_fid, calculate_slice_fid
from training import dataset
from models.stylegan import StyleGAN
import train_stylegan

model_parameters = train_stylegan.ModelParameters()

dataset_name = 'MCB'
if dataset_name == 'MCB':
    tfrecords = ['data/mcb64_screws.tfrecords']
    tf_dataset = dataset.get_mcb_base(tfrecords)
    model_parameters.label_size = 9
elif dataset_name == 'ABC':
    tfrecords = list(Path('data/abc/').rglob('*.tfrecords'))
    tf_dataset = dataset.get_abc_base(tfrecords)

train_dataset = tf_dataset.take(50000).batch(64).prefetch(tf.data.AUTOTUNE)

model = StyleGAN(model_parameters=model_parameters)
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

std_dev_sum = 0
n = 0
for images, labels in train_dataset:
    std_dev = model.discriminator.get_minibatch_statistics(images, labels)
    std_dev_sum += 0
    n += len(images)
real_minibatch_statistics = std_dev_sum / n
print(real_minibatch_statistics)

