
import time, datetime
import os
import json
from dataclasses import dataclass, field, asdict

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from training.visualize import screenshot_and_save
from training import losses
from training import dataset
from models import base_models


class StyleGAN(tf.keras.Model):

    def __init__(self, model_parameters):
        super(StyleGAN, self).__init__()

        self.model_parameters = model_parameters

        self.generator = base_models.GeneratorStatic(
            latent_size=self.model_parameters.latent_size,
            label_size=self.model_parameters.label_size,
            num_layers=self.model_parameters.num_layers,
            base_dim=4,
            img_dim=self.model_parameters.img_dim,
            filters=self.model_parameters.gen_filters)

        self.discriminator = base_models.DiscriminatorStatic(
            base_dim=4,
            img_dim=self.model_parameters.img_dim,
            filters=self.model_parameters.disc_filters,
            label_size=self.model_parameters.label_size)

    def build(self):

        z = tf.zeros(shape=(1, self.model_parameters.latent_size))
        labels = tf.zeros(shape=(1, self.model_parameters.label_size))

        self.discriminator(self.generator(z, labels), labels)

    def compile(self, training_args):
        super(StyleGAN, self).compile()

        self.args = training_args

        self.generator_optimizer = Adam(learning_rate=self.args.gen_lr, beta_1=self.args.adam_beta_1, beta_2=self.args.adam_beta_2, epsilon=self.args.adam_gen_eps)
        self.discriminator_optimizer = Adam(learning_rate=self.args.disc_lr, beta_1=self.args.adam_beta_1, beta_2=self.args.adam_beta_2)

        self.training_metrics = {
            'generator_loss' : tf.keras.metrics.Mean(name='generator_loss'),
            'discriminator_loss' : tf.keras.metrics.Mean(name='discriminator_loss'),
            'real_scores' : tf.keras.metrics.Mean(name='real_scores'),
            'fake_scores' : tf.keras.metrics.Mean(name='fake_scores'),
            'r1_reg' : tf.keras.metrics.Mean(name='r1_reg'),
            'loss_signs_real' : tf.keras.metrics.Mean(name='loss_signs_real'),
            'loss_signs_fake' : tf.keras.metrics.Mean(name='loss_signs_fake'),
        }
        
        self.setup_moving_average()

        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(
                initial_value=1,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                synchronization=tf.VariableSynchronization.ON_READ),
            deception_strength=tf.Variable(
                initial_value=0.,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                synchronization=tf.VariableSynchronization.ON_WRITE),
            seen_images=tf.Variable(
                initial_value=0,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                synchronization=tf.VariableSynchronization.ON_READ),
            generator=self.generator,
            discriminator=self.discriminator,
            generator_ema=self.generator_ema,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer)
            
        self.manager = tf.train.CheckpointManager(
            self.ckpt,
            directory='./tf_ckpts',
            max_to_keep=None)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch")

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_dir = os.path.join('results', current_time)
        os.makedirs(self.results_dir, exist_ok=True)

        with open(os.path.join(self.results_dir, 'model.json'), 'w', encoding='utf-8') as f:
            json.dump(asdict(self.model_parameters), f, ensure_ascii=True, indent=4)
        with open(os.path.join(self.results_dir, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(asdict(self.args), f, ensure_ascii=True, indent=4)

        self.evaluate()

        self.lastblip = time.process_time()

    def setup_moving_average(self):

        self.generator_ema = base_models.GeneratorStatic(
            latent_size=self.model_parameters.latent_size,
            label_size=self.model_parameters.label_size,
            num_layers=self.model_parameters.num_layers,
            base_dim=4,
            img_dim=self.model_parameters.img_dim,
            filters=self.model_parameters.gen_filters)

        z = tf.zeros(shape=(1, self.model_parameters.latent_size))
        labels = tf.zeros(shape=(1, self.model_parameters.label_size))

        self.generator_ema(z, labels)
        self.generator_ema.set_weights(self.generator.get_weights())

    def train_step(self, real_images, real_labels):

        fake_latents = tf.random.normal(shape=(self.args.batch_size_per_replica, self.model_parameters.latent_size))
        # fake_labels = dataset.get_random_labels(batch_size=self.args.batch_size_per_replica, label_size=self.model_parameters.label_size)
        fake_labels = real_labels

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            fake_images = self.generator(
                z=fake_latents,
                labels=fake_labels,
                training=True)
            
            if self.args.apa:
                pseudo_flag = tf.where(
                    tf.random.uniform([self.args.batch_size_per_replica]) < self.ckpt.deception_strength.read_value(),
                    1.,
                    0.)
                if tf.math.reduce_sum(pseudo_flag) > 0:
                    real_images = fake_images * pseudo_flag[..., tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis] + real_images * (1 - pseudo_flag[..., tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis])
                    if self.model_parameters.label_size > 0:
                        real_labels = fake_labels * pseudo_flag[..., tf.newaxis] + real_labels * (1 - pseudo_flag[..., tf.newaxis]) 

            if self.args.r1_gamma != 0 and int(self.ckpt.step.read_value()) % self.args.r1_interval == 0:
                disc_tape.watch(real_images)
                
            fake_output = self.discriminator(
                images=fake_images,
                labels=fake_labels,
                training=True)
            real_output = self.discriminator(
                images=real_images,
                labels=real_labels,
                training=True)
            
            self.training_metrics['real_scores'](real_output)
            self.training_metrics['fake_scores'](fake_output)
            self.training_metrics['loss_signs_real'](tf.math.sign(real_output))
            self.training_metrics['loss_signs_fake'](-tf.math.sign(fake_output))

            gen_loss = losses.generator_logistic_ns(fake_output)
            self.training_metrics['generator_loss'](gen_loss)
            
            disc_loss = losses.discriminator_logistic(real_output, fake_output)
            self.training_metrics['discriminator_loss'](disc_loss)

            reg = tf.zeros((self.args.batch_size_per_replica, 1))

            r1_grads = tf.gradients(tf.math.reduce_sum(real_output), [real_images])[0]
            r1_penalty = tf.math.reduce_sum(tf.math.square(r1_grads), axis=[1,2,3,4])
            r1_penalty = r1_penalty[:, tf.newaxis]
            self.training_metrics['r1_reg'](r1_penalty)
            reg = r1_penalty * (self.args.r1_gamma * 0.5) 

            gen_loss = tf.nn.compute_average_loss(gen_loss, global_batch_size=self.args.global_batch_size)

            disc_loss += reg
            disc_loss = tf.nn.compute_average_loss(disc_loss, global_batch_size=self.args.global_batch_size)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

    def evaluate(self, batch_size: int=16):

        z = tf.random.normal(shape=(batch_size, self.model_parameters.latent_size))
        labels = dataset.get_random_labels(batch_size=batch_size, label_size=self.model_parameters.label_size)

        generated_images_ema = self.generator_ema(
            z=z,
            labels=labels,
            training=False)
        screenshot_and_save(list(generated_images_ema), filepath=os.path.join(self.results_dir, f'step-{int(self.ckpt.step.read_value())}.jpg'))
        
    def update_moving_average(self):
        up_weight = self.generator.get_weights()
        old_weight = self.generator_ema.get_weights()
        new_weights = [self.args.ema_beta * old_weight[i] + (1 - self.args.ema_beta) * up_weight[i] for i in range(len(up_weight))]
        self.generator_ema.set_weights(new_weights)

    def print_info(self):
        print()
        print(f"{'Steps': <31s} {int(self.ckpt.step.read_value())}")  
        print(f"{'train_kimg': <31s} {int(self.ckpt.seen_images.read_value()) // 1000}") 
        print(f"{'Deception Strength': <30s} {self.ckpt.deception_strength.read_value(): 4.5f}") 
        for name, metric in self.training_metrics.items():
            print(f"{name: <30s} {metric.result(): 4.5f}")

        seconds = round((time.process_time() - self.lastblip), 4)
        self.lastblip = time.process_time()

        steps_per_second = self.args.print_interval / seconds
        steps_per_minute = steps_per_second * 60
        steps_per_hour = int(steps_per_minute * 60)

        print(f"{'Steps/Hour': <31s} {steps_per_hour}")

    def checkpoint(self):
        save_path = self.manager.save()
        print("\n Saved checkpoint for step {}: {}".format(int(self.ckpt.step.read_value()), save_path))

    def log_metrics(self):
        if not hasattr(self, 'writer'):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            self.writer = tf.summary.create_file_writer(train_log_dir)

        with self.writer.as_default():
            step = int(self.ckpt.seen_images.read_value() // 1000)
            for name, metric in self.training_metrics.items():
                tf.summary.scalar(name, metric.result(), step=step)
            tf.summary.scalar('deception_strength', self.ckpt.deception_strength.read_value(), step=step)

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()
