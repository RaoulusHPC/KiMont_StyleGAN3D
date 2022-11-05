import argparse
import math
import os
import datetime
import argparse
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from training import dataset, visualize
from optimization.optimizer import LatentOptimizer
from models.base_models import Comparator, LatentMapper
from models.stylegan import StyleGAN
from training import losses
from train_stylegan import ModelParameters


class TrainingArguments:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.1
    train_size: int = 5000
    l2_lambda: float = 2.0

if __name__ == "__main__":

    parameters = TrainingArguments()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lambda0', type=float, default=1.0)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--l2_lambda', type=float, default=2.0)
    parser.add_argument('--real_lambda', type=float, default=0.0)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')

    tfrecords = ['data/latents.tfrecords']
    tf_dataset = dataset.get_projected_dataset(tfrecords)

    tf_dataset = tf_dataset.map(lambda o, g, l, w : (g, w, l))
    train_dataset = tf_dataset.take(parameters.train_size).shuffle(parameters.train_size, reshuffle_each_iteration=True).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf_dataset.skip(parameters.train_size).shuffle(parameters.train_size, reshuffle_each_iteration=True).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)    

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
    ckpt.restore('./tf_ckpts/ckpt-20').expect_partial()   #manager.latest_checkpoint
    generator = model.generator_ema
    generator.trainable = False
    discriminator = model.discriminator
    discriminator.trainable = False

    comparator = Comparator()
    comparator_input = tf.zeros(shape=(1, 64, 64, 64, 1))
    comparator((comparator_input, comparator_input))
    comparator.load_weights('ckpts/comparator/')
    comparator.trainable = False

    mapper = LatentMapper(generator.latent_size, num_layers=3)
    mapper.build((None, generator.latent_size))
    mapper.load_weights('ckpts/mapper/mapper_2.0_1.0_1.0_0.0/20220816-134422/')
    mapper.summary()

    optimizer = Adam(learning_rate=parameters.learning_rate)
    loss_object = tf.losses.BinaryCrossentropy(from_logits=True)

    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    grab0_loss_metric = tf.keras.metrics.Mean(name='grab0_loss')
    grab1_loss_metric = tf.keras.metrics.Mean(name='grab1_loss')
    l2_loss_metric = tf.keras.metrics.Mean(name='l2_loss')
    realism_loss_metric = tf.keras.metrics.Mean(name='realism_loss')

    @tf.function
    def train_step(w, generated_images, l):
        with tf.GradientTape() as tape:
            delta = mapper(w)
            w_opt = w + delta
            optimized_images = generator.synthesize(w_opt)

            comparison_logits0 = comparator((generated_images, optimized_images), training=False)
            grab0_loss = loss_object(tf.zeros_like(comparison_logits0), comparison_logits0)
            comparison_logits1 = comparator((optimized_images, generated_images), training=False)
            grab1_loss = loss_object(tf.ones_like(comparison_logits1), comparison_logits1)
            grab_loss = args.lambda0 * grab0_loss + args.lambda1 * grab1_loss 

            l2_loss = tf.math.reduce_mean(tf.math.square(delta))

            realism_loss = tf.nn.softplus(-discriminator(optimized_images,l))

            loss = grab_loss + args.l2_lambda * l2_loss #+ args.real_lambda * realism_loss

        gradients = tape.gradient(loss, mapper.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mapper.trainable_variables))

        train_loss_metric(loss)

    @tf.function
    def test_step(w, generated_images, l):
        delta = mapper(w)
        w_opt = w + delta
        optimized_images = generator.synthesize(w_opt)

        comparison_logits0 = comparator((generated_images, optimized_images), training=False)
        grab0_loss = loss_object(tf.zeros_like(comparison_logits0), comparison_logits0)
        comparison_logits1 = comparator((optimized_images, generated_images), training=False)
        grab1_loss = loss_object(tf.ones_like(comparison_logits1), comparison_logits1)
        grab_loss = args.lambda0 * grab0_loss + args.lambda1 * grab1_loss

        l2_loss = tf.math.reduce_mean(tf.math.square(delta))

        realism_loss = tf.nn.softplus(-discriminator(optimized_images, l))

        loss = grab_loss + args.l2_lambda * l2_loss  #+ args.real_lambda * realism_loss

        test_loss_metric(loss)
        grab0_loss_metric(grab0_loss)
        grab1_loss_metric(grab1_loss)
        l2_loss_metric(l2_loss)
        realism_loss_metric(realism_loss)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # dir_name = f'{current_time}_{parameters.l2_lambda}_{parameters.lambda0}_{parameters.lambda1}'
    dir_name = f'mapper_{args.l2_lambda}_{args.lambda0}_{args.lambda1}_{args.real_lambda}/{current_time}'
    checkpoint_dir = f'ckpts/mapper/{dir_name}/'
    test_log_dir = f'logs/mapper/{dir_name}'
    test_summary_writer1 = tf.summary.create_file_writer(test_log_dir)

    EPOCHS = parameters.epochs

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss_metric.reset_states()
        test_loss_metric.reset_states()
        grab0_loss_metric.reset_states()
        grab1_loss_metric.reset_states()
        l2_loss_metric.reset_states()
        realism_loss_metric.reset_states()

        for generated_image, w, l  in train_dataset:

            train_step(w, generated_image, l)

        # with train_summary_writer1.as_default():
        #     tf.summary.scalar('loss', train_loss_metric.result(), step=epoch)

        for generated_image, w, l in val_dataset:

            test_step(w, generated_image, l)

        with test_summary_writer1.as_default():
            tf.summary.scalar('test_loss', test_loss_metric.result(), step=epoch+1)
            tf.summary.scalar('grab0_loss', grab0_loss_metric.result(), step=epoch+1)
            tf.summary.scalar('grab1_loss', grab1_loss_metric.result(), step=epoch+1)
            tf.summary.scalar('l2_loss', l2_loss_metric.result(), step=epoch+1)
            tf.summary.scalar('realism_loss', realism_loss_metric.result(), step=epoch+1)
        if epoch == 0 or test_loss_metric.result() < lowest_loss:
            lowest_loss = test_loss_metric.result()
            mapper.save_weights(checkpoint_dir)
            print('Saved weights')

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss_metric.result()}, '
            f'Test Loss: {test_loss_metric.result()} '
        )




