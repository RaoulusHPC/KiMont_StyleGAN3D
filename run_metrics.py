import csv
import tensorflow as tf
from training.visualize import visualize_tensors

from metrics.frechet_inception_distance import calculate_screenshot_fid, calculate_slice_fid
from training import dataset
from models.stylegan import StyleGAN
import train_stylegan

model_parameters = train_stylegan.ModelParameters()

tfrecords = ['data/mcb64_screws.tfrecords']
real_dataset = dataset.get_mcb_dataset(
    tfrecords=tfrecords,
    batch_size=64,
    repeat=1,
    augment_function=None)
    
model = StyleGAN(model_parameters=model_parameters)
model.build()
model.setup_moving_average()

# model.summary()

ckpt = tf.train.Checkpoint(
    seen_images=tf.Variable(0),
    generator_ema=model.generator_ema)
manager = tf.train.CheckpointManager(
    ckpt,
    directory='./tf_ckpts',
    max_to_keep=None)

with open('fids.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for checkpoint in manager.checkpoints[::2]:
        ckpt.restore(checkpoint).expect_partial()
        print(int(ckpt.seen_images // 1000))
        # screenshot_fid = calculate_screenshot_fid(model.mapping_network_ema, model.generator_ema, 10_000, real_dataset, 64)
        slice_fids = calculate_slice_fid(model.generator_ema, 2, real_dataset, 64)
        spamwriter.writerow(list(slice_fids))
