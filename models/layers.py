import math

import numpy as np
import tensorflow as tf

from dnnlib.tflib.ops.my_upfirdn_3d import upsample_conv_3d, conv_downsample_3d, downsample_3d, upsample_3d
from tensorflow.keras.layers import Conv3D, BatchNormalization, Activation


def change_dataformat(layer):
    def mod_call(self, x, *args):
        x = tf.transpose(x, [0, 4, 1, 2, 3])
        x = layer(self, x, *args)
        x = tf.transpose(x, [0, 2, 3, 4, 1])
        return x
    return mod_call

def get_weight_config(weight_shape, lrmul=1.0, gain=1, use_wscale=True):
    fan_in = np.prod(weight_shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    return init_std, runtime_coef


class EqualizedLinear(tf.keras.layers.Layer):
    def __init__(self, units, bias_initializer='zeros', lrmul=1.0, apply_activ=False):
        super(EqualizedLinear, self).__init__()
        self.units = units
        self.bias_initializer = bias_initializer
        self.lrmul = lrmul
        self.apply_activ = apply_activ

    def build(self, input_shape):

        weight_shape = (input_shape[-1], self.units)
        init_std, runtime_coef = get_weight_config(weight_shape, lrmul=self.lrmul)

        self.lrmul = self.add_weight(
            initializer=tf.initializers.Constant(self.lrmul),
            trainable=False,
            name='lr_mul'
        )
        self.runtime_coef = self.add_weight(
            initializer=tf.initializers.Constant(runtime_coef),
            trainable=False,
            name='runtime_coef'
        )
        self.w = self.add_weight(
            shape=weight_shape,
            initializer=tf.initializers.random_normal(0, init_std),
            trainable=True,
            name='linear_weight'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            trainable=True,
            name='linear_bias'
        )

    def call(self, x, adrop_strength=0):
        x = tf.matmul(x, self.w * self.runtime_coef)
        x = adrop(x, adrop_strength, data_format='NDHWC')
        x = x + self.b * self.lrmul
        if self.apply_activ:
            x = tf.nn.leaky_relu(x)
        return x


class EqualizedConv3D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias=True, apply_activ=True, lrmul=1.0):
        super(EqualizedConv3D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.apply_activ = apply_activ
        self.lrmul = lrmul

    def build(self, input_shape):

        weight_shape = (self.kernel_size, self.kernel_size, self.kernel_size, input_shape[-1], self.filters)
        init_std, runtime_coef = get_weight_config(weight_shape, lrmul=self.lrmul)

        self.runtime_coef = self.add_weight(
            initializer=tf.initializers.Constant(runtime_coef),
            trainable=False,
            name='runtime_coef'
        )
        self.w = self.add_weight(
            shape=weight_shape,
            initializer=tf.initializers.random_normal(0, init_std),
            trainable=True,
            name='conv_weight'
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.filters,),
                initializer='zeros',
                trainable=True,
                name='conv_bias'
            )

    def call(self, x, adrop_strength=0):
        weight = self.w * self.runtime_coef

        x = tf.nn.conv3d(x, weight, self.strides, self.padding)
        x = adrop(x, adrop_strength, data_format='NDHWC')

        if self.use_bias:
            x = tf.nn.bias_add(x, self.b)
        if self.apply_activ:
            x = tf.nn.leaky_relu(x)
        return x


class StyleBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, up, name=None):
        super(StyleBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.up = up

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[-1] = self.filters
        if self.up:
            output_shape[1] = output_shape[2] = output_shape[3] = 2 * output_shape[1]
        return output_shape
        
    def build(self, input_shape):
        channel_axis = -1
        input_dim = input_shape[channel_axis]
        self.to_style = EqualizedLinear(input_dim, bias_initializer='ones')
        self.conv = Conv3DWeightModulate(self.filters, kernel_size=self.kernel_size, up=self.up)

        self.b = self.add_weight(shape=(self.filters,),
                                 initializer='zeros',
                                 trainable=True,
                                 name='conv_bias')

    def call(self, x, w):

        # Get style vector s
        style = self.to_style(w)

        # Weight modulated convolution
        x = self.conv(x, style)

        x = tf.nn.bias_add(x, self.b)
        x = tf.nn.leaky_relu(x)

        return x


class ToRGB(tf.keras.layers.Layer):

    def __init__(self, filters=1, kernel_size=1, **kwargs):
        super(ToRGB, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channel_axis = -1
        input_dim = input_shape[channel_axis]

        self.to_style = EqualizedLinear(input_dim, bias_initializer='ones')
        self.conv = Conv3DWeightModulate(self.filters, kernel_size=self.kernel_size, demodulate=False)

        self.b = self.add_weight(shape=(self.filters,),
                                 initializer='zeros',
                                 trainable=True,
                                 name='conv_bias')

    def call(self, x, w):

        style = self.to_style(w)
        
        x = self.conv(x, style)

        x = tf.nn.bias_add(x, self.b)

        return x


class ToRGB2(tf.keras.layers.Layer):

    def __init__(self, filters=1, kernel_size=1, name=None):
        super(ToRGB2, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        
        self.conv = EqualizedConv3D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=[1,1,1,1,1],
            padding='SAME',
            apply_activ=False)

    def call(self, x):

        x = self.conv(x)

        return x


class FromRGB(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=1):
        super(FromRGB, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        weight_shape = (self.kernel_size, self.kernel_size, self.kernel_size, input_shape[-1], self.filters)
        init_std, runtime_coef = get_weight_config(weight_shape, lrmul=1.0)

        self.runtime_coef = self.add_weight(
            initializer=tf.initializers.Constant(runtime_coef),
            trainable=False,
            name='runtime_coef'
        )
        self.w = self.add_weight(
            shape=weight_shape,
            initializer=tf.initializers.random_normal(0, init_std),
            trainable=True,
            name='conv_weight'
        )
        self.b = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            name='conv_bias'
        )

    def call(self, x, adrop_strength=0):
        x = tf.nn.conv3d(x, self.w * self.runtime_coef, data_format='NDHWC', strides=[1,1,1,1,1], padding='SAME')
        x = adrop(x, adrop_strength, data_format='NDHWC')
        x = tf.nn.bias_add(x, self.b)
        x = tf.nn.leaky_relu(x)
        return x


class Conv3DWeightModulate(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size,  up=False, demodulate=True, fused_modconv=False):
        super(Conv3DWeightModulate, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.up = up
        self.demodulate = demodulate
        self.fused_modconv = fused_modconv

    def build(self, input_shape):
        channel_axis = -1
        input_dim = input_shape[channel_axis]
        self.weight_shape = (self.kernel_size, self.kernel_size, self.kernel_size, input_dim, self.filters)

        init_std, self.runtime_coef = get_weight_config(self.weight_shape)

        # Create variable.
        init = tf.initializers.random_normal(0, init_std)
        self.w = self.add_weight(shape=(self.kernel_size, self.kernel_size, self.kernel_size, input_dim, self.filters),
                                initializer=init,
                                trainable=True,
                                name='conv_mod_weights')

    @change_dataformat
    def call(self, x, s):

        s = tf.cast(s, x.dtype)

        w = self.runtime_coef * self.w
        ww = w[tf.newaxis]

        # Modulate.
        ww *= s[:, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]

        if self.demodulate:
            d = tf.math.rsqrt(tf.math.reduce_sum(tf.math.square(ww), axis=[1,2,3,4]) + 1e-8) # [BO] Scaling factor.
            ww *= d[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :] # [BkkkIO] Scale output feature maps.


        if self.fused_modconv:
            w = tf.reshape(tf.transpose(ww, [1, 2, 3, 4, 0, 5]), [ww.shape[1], ww.shape[2], ww.shape[3], ww.shape[4], -1])
            x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3], x.shape[4]])
        else:
            x *= tf.cast(s[:, :, tf.newaxis, tf.newaxis, tf.newaxis], x.dtype)

        if self.up:
            x = UpSampling()(x)
            x = tf.nn.conv3d(x, tf.cast(w, x.dtype), data_format='NCDHW', strides=[1,1,1,1,1], padding='SAME')
        else:
            x = tf.nn.conv3d(x, tf.cast(w, x.dtype), data_format='NCDHW', strides=[1,1,1,1,1], padding='SAME')

        if self.fused_modconv:
            x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3],  x.shape[4]]) # Fused => reshape convolution groups back to minibatch.
        elif self.demodulate:
            x *= tf.cast(d[:, :, tf.newaxis, tf.newaxis, tf.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
        return x


class UpSampling(tf.keras.layers.Layer):

    def __init__(self, k=[1,4,6,4,1], data_format='NCDHW'):
        super(UpSampling, self).__init__()
        self.k = k
        self.data_format = data_format
    
    def call(self, x):
        if self.data_format == 'NDHWC':
            x = tf.transpose(x, [0, 4, 1, 2, 3])
        x = upsample_3d(x, k=self.k)
        if self.data_format == 'NDHWC':
            x = tf.transpose(x, [0, 2, 3, 4, 1])
        return x
    

class DownSampling(tf.keras.layers.Layer):

    def __init__(self):
        super(DownSampling, self).__init__()

    @change_dataformat
    def call(self, x):
        x = downsample_3d(x)
        return x


class DownConv3D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, use_bias=True, apply_activ=True, lrmul=1.0):
        super(DownConv3D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.apply_activ = apply_activ
        self.lrmul = lrmul

    def build(self, input_shape):

        weight_shape = (self.kernel_size, self.kernel_size, self.kernel_size, input_shape[-1], self.filters)
        init_std, runtime_coef = get_weight_config(weight_shape, lrmul=self.lrmul)

        self.runtime_coef = self.add_weight(
            initializer=tf.initializers.Constant(runtime_coef),
            trainable=False,
            name='runtime_coef'
        )
        self.w = self.add_weight(
            shape=weight_shape,
            initializer=tf.initializers.random_normal(0, init_std),
            trainable=True,
            name='conv_down_weight'
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.filters,),
                initializer='zeros',
                trainable=True,
                name='conv_down_bias'
            )

    @change_dataformat
    def call(self, x, adrop_strength=0):
        weight = self.w * self.runtime_coef

        x = conv_downsample_3d(x, weight)
        x = adrop(x, adrop_strength, data_format='NCDHW')

        if self.use_bias:
            x = tf.nn.bias_add(x, self.b, data_format='NCDHW')
        if self.apply_activ:
            x = tf.nn.leaky_relu(x)
        return x


class UpConv3D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, lrmul=1.0):
        super(UpConv3D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.lrmul = lrmul

    def build(self, input_shape):

        weight_shape = (self.kernel_size, self.kernel_size, self.kernel_size, input_shape[-1], self.filters)
        init_std, runtime_coef = get_weight_config(weight_shape, lrmul=self.lrmul)

        self.runtime_coef = self.add_weight(
            initializer=tf.initializers.Constant(runtime_coef),
            trainable=False,
            name='runtime_coef'
        )
        self.w = self.add_weight(
            shape=weight_shape,
            initializer=tf.initializers.random_normal(0, init_std),
            trainable=True,
            name='conv_up_weight'
        )

    @change_dataformat
    def call(self, x):
        x = UpSampling()(x)
        x = tf.nn.conv3d(x, self.w * self.runtime_coef, data_format='NCDHW', strides=[1,1,1,1,1], padding='SAME')
        return x


class MiniBatchStdDev(tf.keras.layers.Layer):

    def __init__(self, group_size=32, num_new_features=1):
        super(MiniBatchStdDev, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    @change_dataformat
    def call(self, x):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        s = x.shape                                             # [NCDHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, self.num_new_features, s[1]//self.num_new_features, s[2], s[3], s[4]])   # [GMNCDHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)                              # [GMNCDHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMNCDHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MNCDHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MNCDHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2,3,4,5], keepdims=True)      # [Mn1111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3], s[4]])             # [NnHWD]  Replicate over group and pixels.
        out = tf.concat([x, y], axis=1) 
        return out


class GeneratorBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters, name=None):
        super(GeneratorBlock, self).__init__(name=name)
        
        self.conv_up = StyleBlock(filters=filters, kernel_size=3, up=True)
        self.conv = StyleBlock(filters=filters, kernel_size=3, up=False)

    def call(self, x, w1, w2):

        x = self.conv_up(x, w1)
        x = self.conv(x, w2)
        
        return x


def adrop(x, adrop_strength, data_format):
    if data_format == 'NCDHW':
        s = [tf.shape(x)[0], x.shape[1]] + [1] * (x.shape.rank - 2)
    elif data_format == 'NDHWC':
        s = [tf.shape(x)[0]] + [1] * (x.shape.rank - 2) + [x.shape[-1]] 
    x *= tf.cast(tf.exp(tf.random.normal(shape=s) * adrop_strength), x.dtype)
    return x


class DiscriminatorBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters1, filters2):
        super(DiscriminatorBlock, self).__init__()

        self.conv = EqualizedConv3D(filters=filters1, kernel_size=3, strides=[1,1,1,1,1], padding='SAME')
        self.conv_down = DownConv3D(filters=filters2, kernel_size=3)

    def call(self, x, adrop_strength=0):

        x = self.conv(x, adrop_strength)
        x = self.conv_down(x, adrop_strength)

        return x


class DiscriminatorResBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters1, filters2):
        super(DiscriminatorResBlock, self).__init__()

        self.conv = EqualizedConv3D(filters=filters1, kernel_size=3, strides=[1,1,1,1,1], padding='SAME')
        self.conv_down = DownConv3D(filters=filters2, kernel_size=3)

        self.residual = DownConv3D(filters=filters2, kernel_size=1, use_bias=False, apply_activ=False)

        self.scale = self.add_weight(
            initializer=tf.initializers.Constant(1. / math.sqrt(2)),
            trainable=False,
            name='res_scale'
        )

    def call(self, x, adrop_strength=0):

        t = x

        # Convolutions
        x = self.conv(x, adrop_strength)
        x = self.conv_down(x, adrop_strength)

        t = self.residual(t, adrop_strength)
        
        # Add the residual and scale
        return (x + t) * self.scale

class ConvBlock(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides, padding, activation):
    super(ConvBlock, self).__init__()
    self.strides = strides
    self.output_filters = filters
    self.conv = Conv3D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
    self.bn = BatchNormalization()
    self.activ = Activation(activation)

  def build(self, input_shape):
    self.input_filters = input_shape[-1]

  def call(self, inputs, training=False):
    x = inputs
    x = self.conv(x)
    x = self.bn(x, training=training)
    x = self.activ(x)
    if self.strides == 1 and self.input_filters == self.output_filters:
      x = x + inputs
    return x