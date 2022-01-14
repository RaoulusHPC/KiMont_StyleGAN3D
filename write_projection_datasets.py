
import argparse
import numpy as np
import tensorflow as tf

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
    
    train_dataset = train_dataset.take(10_000)

    storage = []
    for data, label in train_dataset:
        projector = LatentProjector(model.generator_ema, steps=500, noise=0.01, screenshot=False)
        projector.init_w(label, w_avg_samples=10_000)
        loss, generated_image = projector.project(data)
        if loss < 0.02:
            storage.append((projector.original_image, generated_image, label, projector.w))

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() 
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # Create a dictionary with features that may be relevant.
    def serialize_example(original_image, generated_image, label, w):
        original_image_bytes = tf.io.serialize_tensor(original_image)
        generated_image_bytes = tf.io.serialize_tensor(generated_image)
        label_bytes = tf.io.serialize_tensor(label)
        w_bytes = tf.io.serialize_tensor(w)
        
        feature = {
            'original_image_raw': _bytes_feature(original_image_bytes),
            'generated_image_raw': _bytes_feature(generated_image_bytes),
            'label_raw': _bytes_feature(label_bytes),
            'w_raw': _bytes_feature(w_bytes),
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
    
    record_file = './projected_images.tfrecords'
    options = tf.io.TFRecordOptions(compression_type='GZIP')

    with tf.io.TFRecordWriter(record_file, options) as writer:
        for stored in storage:
            serialized_example = serialize_example(*stored)
            writer.write(serialized_example)