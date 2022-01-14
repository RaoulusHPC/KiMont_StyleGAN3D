from math import pi
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import matplotlib.pyplot as plt 

from training.losses import label_loss
from training.visualize import screenshot_and_save


class OptimizationSchedule(LearningRateSchedule):

    def __init__(self, steps=500, max_learning_rate=0.01, rampup=0.05, rampdown=0.25):
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


class LatentOptimizer(tf.Module):

    def __init__(
        self,
        generator,
        comparator,
        steps=50,
        l2_lambda=0.05,
        save_intermediate_image_every=10,
        filepath=''):

        super(LatentOptimizer, self).__init__()
        self.generator = generator
        self.comparator = comparator
        self.steps = steps
        self.l2_lambda = l2_lambda
        self.save_intermediate_image_every = save_intermediate_image_every
        self.filepath = filepath

    def optimize(self, w):

        self.optimizer = Adam(learning_rate=OptimizationSchedule(steps=self.steps))
        if len(w.shape) == 2:
            w = tf.tile(tf.expand_dims(w, axis=1), [1, self.generator.num_ws, 1])
        self.w_init = tf.convert_to_tensor(w, dtype='float32')
        self.w_opt = tf.Variable(initial_value=self.w_init, trainable=True)

        # Original Image
        self.original_image = self.generator.synthesize(w, broadcast=False)
        self.images = [self.original_image]

        progress_bar = tqdm(range(1, self.steps+1))
        for i in progress_bar:
            generated_image, loss = self.optimization_step()
            progress_bar.set_description(f'loss: {loss.numpy().item():.4f}')

            if self.save_intermediate_image_every > 0 and i % self.save_intermediate_image_every == 0:
                self.images.append(generated_image)

        screenshot_and_save(self.images, filepath=self.filepath, shape=(1, len(self.images)), window_size=(512 * len(self.images), 512))
        changes = tf.where(generated_image - self.original_image > 0, 1., -1.)
        screenshot_and_save([self.original_image, generated_image, changes], filepath='test2.png', shape=(1, 3), window_size=(1536, 512))
        return generated_image, self.w_opt

    @tf.function
    def optimization_step(self):

        with tf.GradientTape() as tape:
            generated_image = self.generator.synthesize(self.w_opt, broadcast=False)
            predicted_label = self.comparator(tf.concat([self.original_image, generated_image], axis=-1), training=False)
            c_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(predicted_label), predicted_label)
            l2_loss = tf.math.reduce_sum(tf.math.square(self.w_opt - self.w_init))
            loss = c_loss + self.l2_lambda * l2_loss
        
        gradient = tape.gradient(loss, self.w_opt)
        self.optimizer.apply_gradients([(gradient, self.w_opt)])
        #tf.print(c_loss, l2_loss)
        return generated_image, loss

    def save_image(self):
        pass

