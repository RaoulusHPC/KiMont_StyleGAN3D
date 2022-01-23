import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Activation, Flatten, InputLayer, Dense, GlobalAveragePooling3D
from tensorflow.python.keras.layers.core import Dropout

from models.layers import FromRGB, GeneratorBlock, StyleBlock, DiscriminatorBlock, ToRGB, ConvBlock, FromRGB, MiniBatchStdDev, EqualizedLinear, EqualizedConv3D, UpSampling, adrop


def normalize_2nd_moment(x, axis=1, eps=1e-8):
    return x * tf.math.rsqrt(tf.math.reduce_mean(tf.math.square(x), axis=axis, keepdims=True) + eps)


class Mapper(tf.keras.Model):

    def __init__(self, num_layers, latent_size):
        super(Mapper, self).__init__()
        self.latent_size = latent_size
    
        self.mapper = Sequential()
        for _ in range(num_layers):
            self.mapper.add(EqualizedLinear(latent_size, lrmul=0.01, apply_activ=True))

    def call(self, x):
        return self.mapper(x)

    def summary(self):
        return self.mapper.summary()


class MappingNetwork(tf.keras.Model):

    def __init__(self, latent_size, label_size, num_layers):
        super(MappingNetwork, self).__init__()
        self.latent_size = latent_size
        self.label_size = label_size
    
        if self.label_size > 0:
            self.embed = EqualizedLinear(units=latent_size)
        
        self.mapper = Sequential()
        for _ in range(num_layers):
            self.mapper.add(EqualizedLinear(latent_size, lrmul=0.01, apply_activ=True))

    def call(self, z, labels):

        x = z
        x = normalize_2nd_moment(x)

        if self.label_size > 0:
            y = self.embed(labels)
            y = normalize_2nd_moment(y)
            x = tf.concat([x, y], axis=1)

        w = self.mapper(x)

        return w


class LatentMapper(tf.keras.Model):

    def __init__(self, latent_size, num_layers):
        super(LatentMapper, self).__init__()
        self.latent_size = latent_size
        
        self.mapper = Sequential()
        for _ in range(num_layers):
            self.mapper.add(EqualizedLinear(latent_size, lrmul=0.01, apply_activ=True))

    def call(self, w):
        #w = normalize_2nd_moment(w)
        delta = self.mapper(w)
        return delta

class LabelGuidedMapper(tf.keras.Model):

    def __init__(self, n_layers, latent_size):
        super(LabelGuidedMapper, self).__init__()
        self.latent_size = latent_size
        
        self.mapper = Mapper(n_layers, latent_size)

        # self.coarse_mapper = Mapper(n_layers, latent_size)
        # self.medium_mapper = Mapper(n_layers, latent_size)
        # self.fine_mapper = Mapper(n_layers, latent_size)

    def call(self, w):
        residual = self.mapper(w)
        return residual + w

    def summary(self):
        return self.mapper.summary()
        

class GeneratorStatic(tf.keras.Model):

    def __init__(self, latent_size, label_size, num_layers, base_dim, img_dim, filters):
        super(GeneratorStatic, self).__init__()

        self.latent_size = latent_size
        self.label_size = label_size
        self.num_layers = num_layers
        self.base_dim = base_dim
        self.img_dim = img_dim

        self.up_times = int(np.log2(self.img_dim // self.base_dim))
        self.num_ws = 2 * (self.up_times + 1)
        
        assert len(filters) == 1 + self.up_times
        self.filters = filters

    def build(self, input_shape):

        self.mapping_network = MappingNetwork(
            latent_size=self.latent_size,
            label_size=self.label_size,
            num_layers=self.num_layers)

        self.initial_constant = self.add_weight(shape=(1, 4, 4, 4, self.filters[0]),
                                                initializer='random_normal',
                                                trainable=True,
                                                name='initial_constant')

        # Generator Blocks
        self.conv = StyleBlock(self.filters[0], kernel_size=3, up=False, name=f'styleblock_res{self.base_dim}')

        # list of generator blocks at increasing resolution
        self.blocks = []
        for i in range(self.up_times):
            res = self.base_dim *  2 ** i
            self.blocks.append(GeneratorBlock(self.filters[i+1], name=f'g_block_res{res}'))

        self.to_rgb = ToRGB(name='to_rgb')
        
        self.final_activation = Activation('tanh', dtype='float32', name='tanh')

    def call(self, z, labels):

        w = self.mapping_network(
            z=z,
            labels=labels)
            
        return self.synthesize(w, broadcast=True)

    def synthesize(self, w, broadcast=True):

        batch_size = tf.shape(w)[0]
        
        if broadcast:
            w = tf.tile(tf.expand_dims(w, axis=1), [1, self.num_ws, 1])

        # Expand the learned constant to match batch size
        x = tf.tile(self.initial_constant, [batch_size, 1, 1, 1, 1])
        x = self.conv(x, w[:, 0])

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, w[:, 2*i+1], w[:, 2*i+2])

        rgb = self.to_rgb(x, w[:, -1])
        x = self.final_activation(rgb)
        return x


class DiscriminatorStatic(tf.keras.Model):

    def __init__(self, base_dim, img_dim, filters, label_size):
        super(DiscriminatorStatic, self).__init__()

        self.base_dim = base_dim
        self.img_dim = img_dim

        self.down_times = int(np.log2(self.img_dim // self.base_dim))

        assert len(filters) == 3 + self.down_times
        self.filters = filters

        self.label_size = label_size

    def build(self, input_shape):

        if self.label_size > 0:
            self.mapping_fmaps = self.filters[-1]
            self.dense = EqualizedLinear(units=self.mapping_fmaps)
        self.from_rgb = FromRGB(self.filters[0])

        self.blocks = []
        for i in range(self.down_times):
            self.blocks.append(DiscriminatorBlock(self.filters[i], self.filters[i+1]))

        self.std_dev = MiniBatchStdDev()
        self.conv = EqualizedConv3D(filters=self.filters[-2], kernel_size=3, strides=[1,1,1,1,1], padding='SAME')
        self.flatten = Flatten()
        self.dense0 = EqualizedLinear(self.filters[-1], apply_activ=True)
        self.dense1 = EqualizedLinear(1 if not hasattr(self, 'mapping_fmaps') else self.mapping_fmaps)

    def call(self, images, labels, adrop_strength=0):

        if self.label_size > 0:
            labels = self.dense(labels)
            labels = normalize_2nd_moment(labels)

        x = self.from_rgb(images, adrop_strength)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, adrop_strength)

        x = self.std_dev(x)
        x = self.conv(x, adrop_strength)
        x = self.flatten(x)
        x = self.dense0(x, adrop_strength)
        x = self.dense1(x)
        if self.label_size > 0:
            x = tf.math.reduce_sum(x * labels, axis=1, keepdims=True) / np.sqrt(self.mapping_fmaps)
        return x


class Comparator(tf.keras.Model):
  def __init__(self):
    super(Comparator, self).__init__()
    self.encoder = tf.keras.Sequential([
      InputLayer(input_shape=(64, 64, 64, 2)),
      ConvBlock(filters=8, kernel_size=3, activation='swish', padding='same', strides=2),
      ConvBlock(filters=16, kernel_size=3, activation='swish', padding='same', strides=2),
      ConvBlock(filters=24, kernel_size=3, activation='swish', padding='same', strides=2),
      ConvBlock(filters=32, kernel_size=3, activation='swish', padding='same', strides=2),
      ConvBlock(filters=32, kernel_size=3, activation='swish', padding='same', strides=1),
    ])
    self.pooling = Flatten()
    self.drop = Dropout(0.5)
    self.dense = Dense(1)

  def call(self, inputs, training=False):
    image1, image2 = inputs
    x = tf.concat([image1, image2], axis=-1)
    x = self.encoder(x, training=training)
    x = self.pooling(x)
    x = self.drop(x, training=training)
    x = self.dense(x)
    return x


