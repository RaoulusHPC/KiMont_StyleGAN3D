# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Custom TensorFlow ops for efficient resampling of 2D images."""

import os
import numpy as np
import tensorflow as tf

#----------------------------------------------------------------------------

def upfirdn_3d(x, k, upx=1, upy=1, upz=1, downx=1, downy=1, downz=1, padx0=0, padx1=0, pady0=0, pady1=0, padz0=0, padz1=0):
    r"""Pad, upsample, FIR filter, and downsample a batch of 2D images.

    Accepts a batch of 2D images of the shape `[majorDim, inH, inW, minorDim]`
    and performs the following operations for each image, batched across
    `majorDim` and `minorDim`:

    1. Pad the image with zeros by the specified number of pixels on each side
       (`padx0`, `padx1`, `pady0`, `pady1`). Specifying a negative value
       corresponds to cropping the image.

    2. Upsample the image by inserting the zeros after each pixel (`upx`, `upy`).

    3. Convolve the image with the specified 2D FIR filter (`k`), shrinking the
       image so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by throwing away pixels (`downx`, `downy`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x:      Input tensor of the shape `[majorDim, inH, inW, minorDim]`.
        k:      2D FIR filter of the shape `[firH, firW]`.
        upx:    Integer upsampling factor along the X-axis (default: 1).
        upy:    Integer upsampling factor along the Y-axis (default: 1).
        downx:  Integer downsampling factor along the X-axis (default: 1).
        downy:  Integer downsampling factor along the Y-axis (default: 1).
        padx0:  Number of pixels to pad on the left side (default: 0).
        padx1:  Number of pixels to pad on the right side (default: 0).
        pady0:  Number of pixels to pad on the top side (default: 0).
        pady1:  Number of pixels to pad on the bottom side (default: 0).
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[majorDim, outH, outW, minorDim]`, and same datatype as `x`.
    """

    return _upfirdn_3d_ref(x=x, k=k, upx=upx, upy=upy, upz=upz, downx=downx, downy=downy, downz=downz, padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1, padz0=padz0, padz1=padz1)

#----------------------------------------------------------------------------

def _upfirdn_3d_ref(x, k, upx, upy, upz, downx, downy, downz, padx0, padx1, pady0, pady1, padz0, padz1):
    """Slow reference implementation of `upfirdn_2d()` using standard TensorFlow ops."""

    x = tf.convert_to_tensor(x)
    k = np.asarray(k, dtype=np.float32)
    assert x.shape.rank == 5
    inH = x.shape[1]
    inW = x.shape[2]
    inD = x.shape[3]
    minorDim = _shape(x, 4)
    kernelH, kernelW, kernelD = k.shape
    assert inW >= 1 and inH >= 1 and inD >= 1
    assert kernelW >= 1 and kernelH >= 1 and kernelD >= 1
    assert isinstance(upx, int) and isinstance(upy, int) and isinstance(upz, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)
    assert isinstance(padz0, int) and isinstance(padz1, int)

    # Upsample (insert zeros).
    x = tf.reshape(x, [-1, inH, 1, inW, inD, minorDim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [-1, inH * upy, inW, inD, minorDim])

    x = tf.reshape(x, [-1, inH * upy, inW, 1, inD, minorDim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, upx - 1], [0, 0], [0, 0]])
    x = tf.reshape(x, [-1, inH * upy, inW * upx, inD, minorDim])

    x = tf.reshape(x, [-1, inH * upy, inW * upx, inD, 1, minorDim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, 0], [0, upz - 1], [0, 0]])
    x = tf.reshape(x, [-1, inH * upy, inW * upx, inD * upz, minorDim])

    # Pad (crop if negative).
    x = tf.pad(x, [[0, 0], [max(pady0, 0), max(pady1, 0)], [max(padx0, 0), max(padx1, 0)], [max(padz0, 0), max(padz1, 0)], [0, 0]])
    x = x[:, max(-pady0, 0) : x.shape[1] - max(-pady1, 0), max(-padx0, 0) : x.shape[2] - max(-padx1, 0), max(-padz0, 0) : x.shape[3] - max(-padz1, 0), :]

    # Convolve with filter.
    x = tf.transpose(x, [0, 4, 1, 2, 3])
    x = tf.reshape(x, [-1, 1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1, inD * upz + padz0 + padz1])
    w = tf.constant(k[::-1, ::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
    x = tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding='VALID', data_format='NCDHW')
    x = tf.reshape(x, [-1, minorDim, inH * upy + pady0 + pady1 - kernelH + 1, inW * upx + padx0 + padx1 - kernelW + 1, inD * upz + padz0 + padz1 - kernelD + 1])
    x = tf.transpose(x, [0, 2, 3, 4, 1])

    # Downsample (throw away pixels).
    return x[:, ::downy, ::downx, ::downz, :]

#----------------------------------------------------------------------------

def upsample_3d(x, k=[1,4,6,4,1], factor=2, gain=1, data_format='NCDHW'):
    r"""Upsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so that
    if the input pixels are constant, they will be scaled by the specified `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded with
    zeros so that its shape is a multiple of the upsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to nearest-neighbor
                      upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]` or
        `[N, H * factor, W * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1

    k = _setup_kernel(k) * (gain * (factor ** 3))
    p = k.shape[0] - factor

    #TODO
    pad0 = (p+1)//2+factor-1
    pad1 = p//2

    # pad02 = (k.shape[0] + factor - 1) // 2
    # pad12 = (k.shape[0] - factor) // 2

    return _simple_upfirdn_3d(x, k, up=factor, pad0=pad0, pad1=pad1, data_format=data_format)

#----------------------------------------------------------------------------

def downsample_3d(x, k=[1,4,6,4,1], factor=2, gain=1, data_format='NCDHW'):
    r"""Downsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized so that
    if the input pixels are constant, they will be scaled by the specified `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded with
    zeros so that its shape is a multiple of the downsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return _simple_upfirdn_3d(x, k, down=factor, pad0=(p+1)//2, pad1=p//2, data_format=data_format)

#----------------------------------------------------------------------------

def upsample_conv_3d(x, w, k=[1,4,6,4,1], factor=2, gain=1, data_format='NCDHW'):
    r"""Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`.
                      Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to nearest-neighbor
                      upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]` or
        `[N, H * factor, W * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    w = tf.convert_to_tensor(w)
    assert w.shape.rank == 5
    convH = w.shape[0]
    convW = w.shape[1]
    convD = w.shape[2]
    inC = _shape(w, 3)
    outC = _shape(w, 4)
    assert convW == convH and convW == convD

    # Setup filter kernel.
    k = _setup_kernel(k) * (gain * (factor ** 3))
    p = (k.shape[0] - factor) - (convW - 1)

    # Determine data dimensions.
    if data_format == 'NCDHW':
        stride = [1, 1, factor, factor, factor]
        output_shape = [_shape(x, 0), outC, (_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW, (_shape(x, 4) - 1) * factor + convD]
        num_groups = _shape(x, 1) // inC
    else:
        stride = [1, factor, factor, factor, 1]
        output_shape = [_shape(x, 0), (_shape(x, 1) - 1) * factor + convH, (_shape(x, 2) - 1) * factor + convW, (_shape(x, 3) - 1) * factor + convD, outC]
        num_groups = _shape(x, 4) // inC

    # Transpose weights.
    w = tf.reshape(w, [convH, convW, convD, inC, num_groups, -1])
    w = tf.transpose(w[::-1, ::-1, ::-1], [0, 1, 2, 5, 4, 3])
    w = tf.reshape(w, [convH, convW, convD, -1, num_groups * inC])

    # Execute.
    x = tf.nn.conv3d_transpose(x, w, output_shape=output_shape, strides=stride, padding='VALID', data_format=data_format)
    return _simple_upfirdn_3d(x, k, pad0=(p+1)//2+factor-1, pad1=p//2+1, data_format=data_format)

#----------------------------------------------------------------------------

def conv_downsample_3d(x, w, k=[1,4,6,4,1], factor=2, gain=1, data_format='NCDHW'):
    r"""Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`.
                      Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    w = tf.convert_to_tensor(w)
    convH, convW, convD, _inC, _outC = w.shape.as_list()
    assert convW == convH and convH == convD

    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    if data_format == 'NCDHW':
        s = [1, 1, factor, factor, factor]
    else:
        s = [1, factor, factor, factor, 1]
    x = _simple_upfirdn_3d(x, k, pad0=(p+1)//2, pad1=p//2, data_format=data_format)
    return tf.nn.conv3d(x, w, strides=s, padding='VALID', data_format=data_format)

#----------------------------------------------------------------------------
# Internal helper funcs.

def _shape(tf_expr, dim_idx):
    if tf_expr.shape.rank is not None:
        dim = tf_expr.shape[dim_idx]
        if dim is not None:
            return dim
    return tf.shape(tf_expr)[dim_idx]

def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)

    if k.ndim == 1:
        k_tdot = np.tensordot(k, k, axes=0)
        k = np.tensordot(k_tdot, k, axes=0)

    k = np.divide(k, np.sum(k))
    assert k.ndim == 3
    assert k.shape[0] == k.shape[1] and k.shape[ 0 ] == k.shape[ 2 ]
    return k


def _simple_upfirdn_3d(x, k, up=1, down=1, pad0=0, pad1=0, data_format='NCDHW'):
    assert data_format in ['NCDHW']
    assert x.shape.rank == 5
    y = x
    if data_format == 'NCDHW':
        y = tf.reshape(y, [-1, _shape(y, 2), _shape(y, 3), _shape(y, 4), 1])
    y = upfirdn_3d(y, k, upx=up, upy=up, upz=up, downx=down, downy=down, downz=down, padx0=pad0, padx1=pad1, pady0=pad0, pady1=pad1, padz0=pad0, padz1=pad1)
    if data_format == 'NCDHW':
        y = tf.reshape(y, [-1, _shape(x, 1), _shape(y, 1), _shape(y, 2), _shape(y, 3)])
    return y

#----------------------------------------------------------------------------
