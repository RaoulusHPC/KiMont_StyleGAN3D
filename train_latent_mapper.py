import argparse
import math
import os
import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from training import dataset, visualize
from optimization.optimizer import LatentOptimizer
from models.base_models import Comparator, LatentMapper
from models.stylegan import StyleGAN
from train_stylegan import ModelParameters


class TrainingArguments:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.5
    train_size: int = 5000
    l2_lambda: float = 1.0
    lambda0: float = 0.5
    lambda1: float = 0.5

if __name__ == "__main__":

    parameters = TrainingArguments()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')

    tfrecords = ['data/projected_images.tfrecords']
    tf_dataset = dataset.get_projected_dataset(tfrecords)
    # tf_dataset = tf_dataset.take(600)
    tf_dataset = tf_dataset.map(lambda o, g, l, w : (g, w))
    train_dataset = tf_dataset.take(parameters.train_size).shuffle(5000, reshuffle_each_iteration=True).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf_dataset.skip(parameters.train_size).shuffle(5000, reshuffle_each_iteration=True).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)    
    
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
        directory='ckpts/stylegan/',
        max_to_keep=None)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    generator = model.generator_ema
    generator.trainable = False

    # visualize.visualize_tensors([original_image, generated_image, binarize(generated_image)], shape=(1, 3))
    comparator = Comparator()
    comparator_input = tf.zeros(shape=(1, 64, 64, 64, 1))
    comparator((comparator_input, comparator_input))
    comparator.load_weights('ckpts/comparator/')
    comparator.trainable = False

    mapper = LatentMapper(generator.latent_size, num_layers=4)
    mapper.build((None, generator.latent_size))
    mapper.summary()

    optimizer = Adam(learning_rate=parameters.learning_rate)
    loss_object = tf.losses.BinaryCrossentropy(from_logits=True)

    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    grab0_loss_metric = tf.keras.metrics.Mean(name='grab0_loss')
    grab1_loss_metric = tf.keras.metrics.Mean(name='grab1_loss')
    l2_loss_metric = tf.keras.metrics.Mean(name='l2_loss')

    @tf.function
    def train_step(w, generated_image):
        with tf.GradientTape() as tape:
            delta = mapper(w)
            w_opt = w + delta
            optimized_image = generator.synthesize(w_opt)
            comparison_logits = comparator((generated_image, optimized_image), training=False)
            grab0_loss = loss_object(tf.zeros_like(comparison_logits), comparison_logits)
            comparison_logits2 = comparator((optimized_image, generated_image), training=False)
            grab1_loss = loss_object(tf.ones_like(comparison_logits2), comparison_logits2)
            grab_loss = parameters.lambda0 * grab0_loss + parameters.lambda1 * grab1_loss 
            l2_loss = tf.math.reduce_mean(tf.math.square(delta))
            loss = grab_loss + parameters.l2_lambda * l2_loss
        gradients = tape.gradient(loss, mapper.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mapper.trainable_variables))

        train_loss_metric(loss)

    @tf.function
    def test_step(w, generated_image):
        delta = mapper(w)
        w_opt = w + delta
        optimized_image = generator.synthesize(w_opt)
        comparison_logits = comparator((generated_image, optimized_image), training=False)
        grab0_loss = loss_object(tf.zeros_like(comparison_logits), comparison_logits)
        comparison_logits2 = comparator((optimized_image, generated_image), training=False)
        grab1_loss = loss_object(tf.ones_like(comparison_logits2), comparison_logits2)
        grab_loss = parameters.lambda0 * grab0_loss + parameters.lambda1 * grab1_loss 
        l2_loss = tf.math.reduce_mean(tf.math.square(delta))
        loss = grab_loss + parameters.l2_lambda * l2_loss

        test_loss_metric(loss)
        grab0_loss_metric(grab0_loss)
        grab1_loss_metric(grab1_loss)
        l2_loss_metric(l2_loss)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f'{current_time}_{parameters.lambda0}_{parameters.lambda1}'
    checkpoint_dir = f'ckpts/mapper/{dir_name}/'
    train_log_dir = f'logs/mapper/{dir_name}/train'
    test_log_dir = f'logs/mapper/{dir_name}/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    EPOCHS = parameters.epochs

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss_metric.reset_states()
        test_loss_metric.reset_states()

        for generated_image, w  in train_dataset:
            train_step(w, generated_image)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_metric.result(), step=epoch)

        for generated_image, w in val_dataset:
            test_step(w, generated_image)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss_metric.result(), step=epoch)
            tf.summary.scalar('grab0_loss', grab0_loss_metric.result(), step=epoch)
            tf.summary.scalar('grab1_loss', grab1_loss_metric.result(), step=epoch)
            tf.summary.scalar('l2_loss', l2_loss_metric.result(), step=epoch)
            if epoch == 0 or test_loss_metric.result() < lowest_loss:
                lowest_loss = test_loss_metric.result()
                mapper.save_weights(checkpoint_dir)
                print('Saved weights')

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss_metric.result()}, '
            f'Test Loss: {test_loss_metric.result()}'
        )



    