from math import pi
from tensorflow.python.ops import unconnected_gradients
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import numpy as np
import matplotlib.pyplot as plt 

from training.visualize import screenshot_and_save


class ProjectionSchedule(LearningRateSchedule):
    def __init__(self, steps=1000, max_learning_rate=0.1, rampup=0.05, rampdown=0.25):
        self.steps = steps
        self.max_learning_rate = max_learning_rate
        self.rampup = rampup
        self.rampdown = rampdown

    def plot(self):
        learning_rates = [self(step) for step in range(1, self.steps+1)] 
        plt.plot(learning_rates)
        plt.show()

    def __call__(self, step):
        t = step / self.steps
        lr_ramp = tf.math.minimum(1., (1. - t) / self.rampdown)
        lr_ramp = 0.5 - 0.5 * tf.math.cos(lr_ramp * pi)
        lr_ramp = lr_ramp * tf.math.minimum(1., t / self.rampup)

        return self.max_learning_rate * lr_ramp


class LatentProjector(tf.Module):

    def __init__(
        self,
        generator,
        space='W+',
        noise=0.05,
        noise_ramp=0.75,
        steps=1000,
        screenshot=False):

        super(LatentProjector, self).__init__()

        self.generator = generator
        self.space = space
        self.noise = noise
        self.noise_ramp = noise_ramp
        self.steps = steps
        self.screenshot = screenshot

    def init_w(self, label, w_avg_samples=10_000):
        z = tf.random.normal(shape=(w_avg_samples, self.generator.latent_size))
        label = tf.tile(label, [w_avg_samples, 1])
        w = self.generator.mapping_network(z, label)
        self.w_avg = np.mean(w, axis=0, keepdims=True)
        self.w_std = (np.sum((w - self.w_avg) ** 2) / w_avg_samples) ** 0.5

    def get_noise(self, step):
        t = step / self.steps
        noise_strength = self.w_std * self.noise * max(0, 1 - t / self.noise_ramp) ** 2
        noise = tf.random.normal(self.w.shape) * noise_strength
        return noise

    def project(self, image):
        
        self.optimizer = Adam(learning_rate=ProjectionSchedule(steps=self.steps))
        self.original_image = tf.convert_to_tensor(image, dtype='float32')
        # initial_value = tf.tile(tf.expand_dims(self.w_avg, axis=1), [1, self.generator.num_ws, 1])
        # self.w = tf.Variable(initial_value=initial_value)
        self.w = tf.Variable(initial_value=self.w_avg, trainable=True)
        w_expanded = tf.tile(tf.expand_dims(self.w_avg, axis=1), [1, self.generator.num_ws, 1])
        self.delta = tf.Variable(initial_value=0 * w_expanded, trainable= self.space == 'W+')

        progress_bar = tqdm(range(1, self.steps+1))
        for step in progress_bar:
            noise = self.get_noise(step)
            generated_image, loss = self.projection_step(noise)
            progress_bar.set_description(f'loss: {loss.numpy().item():.4f}')
            # if step % 50 == 1:
            #     screenshot_and_save([self.original_image, generated_image], filepath=f'projection/results/projection{step}.png', shape=(1, 2), window_size=1000)
        
        if self.screenshot:
            screenshot_and_save([self.original_image, generated_image], filepath='projection/results/projection.png', shape=(1, 2), window_size=1000)
        
        return loss, generated_image

    @tf.function
    def projection_step(self, noise):

        with tf.GradientTape() as tape:
            w_noised = self.w + noise
            w_noised = tf.tile(tf.expand_dims(w_noised, axis=1), [1, self.generator.num_ws, 1])
            w_noised += self.delta
            generated_image = self.generator.synthesize(w_noised, broadcast=False)
            loss = tf.math.reduce_mean(tf.math.square(generated_image - self.original_image))
            loss += tf.math.reduce_sum(tf.math.square(self.delta))
            # if self.space == 'W+':
            #     w_noised  == tf.tile(tf.expand_dims(w_noised, axis=1), [1, self.generator.num_ws, 1])
            #     w_noised += self.delta
            #     generated_image = self.generator.synthesize(w_noised, broadcast=False)
            #     loss = tf.math.reduce_mean(tf.math.square(generated_image - self.original_image))
            #     loss += tf.math.reduce_sum(tf.math.square(self.delta))
            # else:
            #     generated_image = self.generator.synthesize(w_noised)
            #     loss = tf.math.reduce_mean(tf.math.square(generated_image - self.original_image))

        gradients = tape.gradient(loss, [self.w, self.delta], unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.optimizer.apply_gradients(zip(gradients, [self.w, self.delta]))

        return generated_image, loss

