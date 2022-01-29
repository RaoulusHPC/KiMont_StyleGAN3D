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
        steps=200,
        lr=0.01,
        lambda0=1.0,
        lambda1=0.0,
        l2_lambda=0.5,
        save_intermediate_image_every=10,
        filepath=''):

        super(LatentOptimizer, self).__init__()
        self.generator = generator
        self.comparator = comparator
        self.steps = steps
        self.lr = lr
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.l2_lambda = l2_lambda
        self.save_intermediate_image_every = save_intermediate_image_every
        self.filepath = filepath

    def optimize(self, w):

        self.optimizer = Adam(learning_rate=OptimizationSchedule(steps=self.steps, max_learning_rate=self.lr))
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        if len(w.shape) == 2:
            w = tf.tile(tf.expand_dims(w, axis=1), [1, self.generator.num_ws, 1])
        self.w_init = tf.convert_to_tensor(w, dtype='float32')
        self.w_opt = tf.Variable(initial_value=self.w_init, trainable=True)

        # Original Image
        self.original_image = self.generator.synthesize(w, broadcast=False)
        self.images = [self.original_image]

        progress_bar = tqdm(range(1, self.steps+1))
        for i in progress_bar:
            optimized_image, loss = self.optimization_step()
            progress_bar.set_description(f'loss: {loss.numpy().item():.4f}')

            if self.save_intermediate_image_every > 0 and i % self.save_intermediate_image_every == 0:
                self.images.append(optimized_image)

        #screenshot_and_save(self.images, filepath=self.filepath, shape=(1, len(self.images)), window_size=(512 * len(self.images), 512))
        #changes = tf.where((optimized_image - self.original_image) > 0, 1., -1.)
        #screenshot_and_save([self.original_image, optimized_image, changes], filepath=self.filepath, shape=(1, 3), window_size=(3000, 1000))
        optimized_image = tf.where(optimized_image > 0, 1.0, -1.0)
        return optimized_image, self.w_opt

    @tf.function
    def optimization_step(self):

        with tf.GradientTape() as tape:
            optimized_image = self.generator.synthesize(self.w_opt, broadcast=False)

            comparison_logits0 = self.comparator((self.original_image, optimized_image), training=False)
            grab0_loss = self.loss_object(tf.zeros_like(comparison_logits0), comparison_logits0)

            comparison_logits1 = self.comparator((optimized_image, self.original_image), training=False)
            grab1_loss = self.loss_object(tf.ones_like(comparison_logits1), comparison_logits1)
            
            grab_loss = self.lambda0 * grab0_loss + self.lambda1 * grab1_loss 
            l2_loss = tf.math.reduce_sum(tf.math.square(self.w_opt - self.w_init))
            loss = grab_loss + self.l2_lambda * l2_loss

        gradient = tape.gradient(loss, self.w_opt)
        self.optimizer.apply_gradients([(gradient, self.w_opt)])
        #tf.print(c_loss, l2_loss)
        return optimized_image, loss

    def save_image(self):
        pass

