# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.

This script requires TFC v2 (`pip install tensorflow-compression==2.*`).
"""

import argparse
import glob
import sys
import os
from time import time
from absl import app
from absl.flags import argparse_flags
from numpy.core.numeric import True_
import tensorflow as tf
from tensorflow._api.v2 import image
from tensorflow.python.ops.gen_array_ops import shape
#import tensorflow_datasets as tfds
import tensorflow_compression as tfc
from timeit import default_timer as timer
from datetime import datetime
import time
import cv2
import numpy as np

tf.compat.v1.enable_eager_execution()
#for gpu
#
#config=tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth=True
#sess=tf.compat.v1.Session(config=config)
#
def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)


def padd_image(image):
  #image=cv2.imread(filename)
  height=image.shape[0]
  width=image.shape[1]
  wr=width%16
  hr=height%16
  
  if wr!=0:
    wr=16-wr
  if hr!=0:
    hr=16-hr

  p_height=height+hr
  p_width=width+wr
  
  dst = cv2.copyMakeBorder(image, 0, hr, 0, wr, cv2.BORDER_REPLICATE, None)
  #image= cv2.resize(image, (p_width,p_height))
  image=dst.astype(np.uint8)
  #image /=255
  
  return image


class Back_1(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super().__init__()
    self.back1=tfc.SignalConv2D(
            3, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True))
  def call(self, tensor):
    out=self.back1(tensor)
    return out


class Back_2(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super().__init__()

    self.back1=tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True))
    self.back2=  tfc.SignalConv2D(
            3, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True))
 

  def call(self, tensor):
    tensor=self.back1(tensor)
    tensor=self.back2(tensor)
    return tensor


class Back_3(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super().__init__()

    self.back1=tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True))
    self.back2=  tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True))
    self.back3=  tfc.SignalConv2D(
            3, (5, 5), name="layer_2", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_2", inverse=True))
   
    

  def call(self, tensor):
    tensor=self.back1(tensor)
    tensor=self.back2(tensor)
    tensor=self.back3(tensor)
  
    return tensor

class Back_4(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super().__init__()

    self.back1=tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True))
    self.back2=  tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True))
    self.back3=  tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_2", inverse=True))
    self.back4= tfc.SignalConv2D(
            3, (5, 5), name="layer_3", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=None)
    
 

  def call(self, tensor):
    tensor=self.back1(tensor)
    tensor=self.back2(tensor)
    tensor=self.back3(tensor)
    tensor=self.back4(tensor)
    return tensor

class ErrorDown(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super().__init__()
    self.conv1=tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0"))
    self.conv2=tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1"))
    self.conv3= tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_2"))
    self.conv4=tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=None)
    
   

  def call(self, tensor):
    tensor=self.conv1(tensor)
    tensor=self.conv2(tensor)
    tensor=self.conv3(tensor)
    tensor=self.conv4(tensor)
    return tensor


class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters):
    super().__init__(name="analysis")
    self.lmda=tf.keras.layers.Lambda(lambda x: x / 255.)
    self.conv1=tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0"))

    self.deconv1=Back_1(num_filters)

    self.conv2=tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1"))

    self.deconv2=Back_2(num_filters)

    self.conv3=tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_2"))

    self.deconv3=Back_3(num_filters)

    self.conv4=tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=None)

    self.deconv4=Back_4(num_filters)
    self.errdown=ErrorDown(num_filters)
  
  def call(self, tensor):
    tensor=self.lmda(tensor)
    out=self.conv1(tensor)
    back1=self.deconv1(out)
    error1=tensor-back1

    out=self.conv2(out)
    back2=self.deconv2(out)
    error2=tensor-back2

    out=self.conv3(out)
    back3=self.deconv3(out)
    error3=tensor-back3

    out=self.conv4(out)
    back4=self.deconv4(out)
    error4=tensor-back4

    error=tf.concat([error1,error2,error3,error4],-1)
    error=self.errdown(error)

    out=out+error

    return out


class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters):
    super().__init__(name="synthesis")
    self.deconv1=tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True))
    self.deconv2=tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True))
    self.deconv3=tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True))
    self.deconv4=tfc.SignalConv2D(
        3, (5, 5), name="layer_3", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=None)
    self.lmda=tf.keras.layers.Lambda(lambda x: x * 255.)

  def call(self, tensor):
    tensor=self.deconv1(tensor)
    tensor=self.deconv2(tensor)
    tensor=self.deconv3(tensor)
    tensor=self.deconv4(tensor)
    tensor=self.lmda(tensor)
    return tensor

#class AnalysisTransform(tf.keras.Sequential):
#  """The analysis transform."""
#
#  def __init__(self, num_filters):
#    super().__init__(name="analysis")
#    self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
#    self.add(tfc.SignalConv2D(
#        num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
#        padding="same_zeros", use_bias=True,
#        activation=tfc.GDN(name="gdn_0")))
#    self.add(tfc.SignalConv2D(
#        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
#        padding="same_zeros", use_bias=True,
#        activation=tfc.GDN(name="gdn_1")))
#    self.add(tfc.SignalConv2D(
#        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
#        padding="same_zeros", use_bias=True,
#        activation=tfc.GDN(name="gdn_2")))
#    self.add(tfc.SignalConv2D(
#        num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
#        padding="same_zeros", use_bias=True,
#        activation=None))
#
#
#class SynthesisTransform(tf.keras.Sequential):
#  """The synthesis transform."""
#
#  def __init__(self, num_filters):
#    super().__init__(name="synthesis")
#    self.add(tfc.SignalConv2D(
#        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
#        padding="same_zeros", use_bias=True,
#        activation=tfc.GDN(name="igdn_0", inverse=True)))
#    self.add(tfc.SignalConv2D(
#        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
#        padding="same_zeros", use_bias=True,
#        activation=tfc.GDN(name="igdn_1", inverse=True)))
#    self.add(tfc.SignalConv2D(
#        num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
#        padding="same_zeros", use_bias=True,
#        activation=tfc.GDN(name="igdn_2", inverse=True)))
#    self.add(tfc.SignalConv2D(
#        1, (5, 5), name="layer_3", corr=False, strides_up=2,
#        padding="same_zeros", use_bias=True,
#        activation=None))
#    self.add(tf.keras.layers.Lambda(lambda x: x * 255.))


class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_analysis")
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))


class HyperSynthesisTransform(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=None))


class BMSHJ2018Model(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda, num_filters, num_scales, scale_min, scale_max):
    super().__init__()
    self.lmbda = lmbda
    self.num_scales = num_scales
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    self.analysis_transform = AnalysisTransform(num_filters)
    self.synthesis_transform = SynthesisTransform(num_filters)
    self.hyper_analysis_transform = HyperAnalysisTransform(num_filters)
    self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters)
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
    self.build((None, None, None, 3))

  def call(self, x, training):
    """Computes rate and distortion losses."""
    entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=False)
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False)

    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))
    z_hat, side_bits = side_entropy_model(z, training=training)
    indexes = self.hyper_synthesis_transform(z_hat)
    y_hat, bits = entropy_model(y, indexes, training=training)
    x_hat = self.synthesis_transform(y_hat)

    # Total number of bits divided by total number of pixels.
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
    # Mean squared error across pixels.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))

    # The rate-distortion Lagrangian.
    loss = bpp + self.lmbda * mse
    return loss, bpp, mse

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def test_step(self, x):
    loss, bpp, mse = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")
    

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=True)
    self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
  ])
  def compress(self, x):
    """Compresses an image."""
    # Add batch dimension and cast to float.

    x = tf.expand_dims(x, 0)
    x = tf.cast(x, dtype=tf.float32)
    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))
    # Preserve spatial shapes of image and latents.
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    z_shape = tf.shape(z)[1:-1]
    z_hat, _ = self.side_entropy_model(z, training=False)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    side_string = self.side_entropy_model.compress(z)
    string = self.entropy_model.compress(y, indexes)
    return string, side_string, x_shape, y_shape, z_shape

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
  ])
  def decompress(self, string, side_string, x_shape, y_shape, z_shape):
    """Decompresses an image."""
    z_hat = self.side_entropy_model.decompress(side_string, z_shape)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    y_hat = self.entropy_model.decompress(string, indexes)
    x_hat = self.synthesis_transform(y_hat)
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to 8-bit integer.
    return tf.saturate_cast(tf.round(x_hat), tf.uint8)


def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
  image = tf.image.random_crop(image, (patchsize, patchsize, 3))
  return tf.cast(image, tf.float32)


def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  #with tf.device("/cpu:0"):
  with tf.device("/gpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patchsize))
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def get_custom_dataset(split, args):
  """Creates input data pipeline from custom PNG images."""
  #with tf.device("/cpu:0"):
  with tf.device("/gpu:0"):
  #strategy=tf.distribute.MirroredStrategy()
  #with strategy.scope():
    files = glob.glob(args.train_glob)
    if not files:
      raise RuntimeError(f"No training images found with glob "
                         f"'{args.train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: crop_image(read_png(x), args.patchsize),
        num_parallel_calls=args.preprocess_threads)
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def train(args):
  """Instantiates and trains the model."""
  if args.check_numerics:
    tf.debugging.enable_check_numerics()

  model = BMSHJ2018Model(
      args.lmbda, args.num_filters, args.num_scales, args.scale_min,
      args.scale_max)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
  )

  if args.train_glob:
    train_dataset = get_custom_dataset("train", args)
    validation_dataset = get_custom_dataset("validation", args)
  else:
    train_dataset = get_dataset("clic", "train", args)
    validation_dataset = get_dataset("clic", "validation", args)
  validation_dataset = validation_dataset.take(args.max_validation_steps)

  model.fit(
      train_dataset.prefetch(8),
      epochs=args.epochs,
      steps_per_epoch=args.steps_per_epoch,
      validation_data=validation_dataset.cache(),
      validation_freq=1,
      callbacks=[
          tf.keras.callbacks.TerminateOnNaN(),
          tf.keras.callbacks.TensorBoard(
              log_dir=args.train_path,
              histogram_freq=1, update_freq="epoch"),
          tf.keras.callbacks.experimental.BackupAndRestore(args.train_path),
      ],
      verbose=int(args.verbose),
  )
  model.save(args.model_path)


def compress(args):
  with tf.device("/cpu:0"):
    """Compresses an image."""
    # Load model and use it to compress the image.
    model = tf.keras.models.load_model(args.model_path)
    #x = read_png(args.input_file)
    xx = cv2.imread(args.input_file,cv2.IMREAD_UNCHANGED)
    xx= cv2.cvtColor(xx, cv2.COLOR_BGR2RGB)
    x=padd_image(xx)

    xx = tf.expand_dims(xx, 0)

    #tensors= model.compress(x)
    string, side_string, x_shape, y_shape, z_shape = model.compress(x)

    x_shape = tf.shape(xx)[1:-1]

    tensors=[string,side_string,x_shape,y_shape,z_shape]

    # Write a binary file with the shape information and the compressed string.
    packed = tfc.PackedTensors()
    packed.pack(tensors)
    with open(args.output_file, "wb") as f:
      f.write(packed.string)

    # If requested, decompress the image and measure performance.
    if args.verbose:
      x_hat = model.decompress(*tensors)

      # Cast to float in order to compute metrics.
      x = tf.cast(x, tf.float32)
      x_hat = tf.cast(x_hat, tf.float32)
      mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
      psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
      msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
      msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

      # The actual bits per pixel including entropy coding overhead.
      num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
      bpp = len(packed.string) * 8 / num_pixels

      print(f"Mean squared error: {mse:0.4f}")
      print(f"PSNR (dB): {psnr:0.2f}")
      print(f"Multiscale SSIM: {msssim:0.4f}")
      print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
      print(f"Bits per pixel: {bpp:0.4f}")


def decompress(args):
  with tf.device("/cpu:0"):
    """Decompresses an image."""

    image=cv2.imread(args.original, cv2.IMREAD_UNCHANGED)
    height=image.shape[0]
    width=image.shape[1]

    # Load the model and determine the dtypes of tensors required to decompress.
    model = tf.keras.models.load_model(args.model_path)
    dtypes = [t.dtype for t in model.decompress.input_signature]

    # Read the shape information and compressed string from the binary file,
    # and decompress the image using the model.
    with open(args.input_file, "rb") as f:
      packed = tfc.PackedTensors(f.read())
    tensors = packed.unpack(dtypes)
    x_hat = model.decompress(*tensors)

    #h=x_hat.shape[0]
    #w=x_hat.shape[1]
#
    #rh=h%16
    #rw=w%16
#
    #if rh!=0:
    #    rh=16-rh
    #if rw!=0:
    #    rw=16-rw
#
    #unp_height=h-rh
    #unp_width=w-rw
#
    #x_hat=tf.reshape(x_hat,[height,width,3])

    # Write reconstructed image out as a PNG file.
    write_png(args.output_file, x_hat)


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training or compressing.")
  parser.add_argument(
      "--model_path", default="bmshj2018",
      help="Path where to save/load the trained model.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--train_glob", type=str, default=None,
      help="Glob pattern identifying custom training data. This pattern must "
           "expand to a list of RGB images in PNG format. If unspecified, the "
           "CLIC dataset from TensorFlow Datasets is used.")
  train_cmd.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters per layer.")
  train_cmd.add_argument(
      "--num_scales", type=int, default=64,
      help="Number of Gaussian scales to prepare range coding tables for.")
  train_cmd.add_argument(
      "--scale_min", type=float, default=.11,
      help="Minimum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--scale_max", type=float, default=256.,
      help="Maximum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--train_path", default="/tmp/train_bmshj2018",
      help="Path where to log training metrics for TensorBoard and back up "
           "intermediate model checkpoints.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training and validation.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=1000,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--max_validation_steps", type=int, default=16,
      help="Maximum number of batches to use for validation. If -1, use one "
           "patch from each image in the training set.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  train_cmd.add_argument(
      "--check_numerics", action="store_true",
      help="Enable TF support for catching NaN and Inf in tensors.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help=f"Output filename (optional). If not provided, appends '{ext}' to "
             f"the input filename.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  now = datetime.now()
  print("Start date and time: ", now.strftime("%d/%m/%Y %H:%M:%S"))
  start = timer()
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)
  end = timer()
  print("Time consumed = ", end-start)

  now = datetime.now()
  print("End date and time: ", now.strftime("%d/%m/%Y %H:%M:%S"))

if __name__ == "__main__":

  # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
  
  # if tf.config.list_physical_devices('GPU'):
  #   print('yes')
  # else:
  #   print('no')

  # if tf.test.is_gpu_available():
  #   print('yes')
  # else:
  #   print('no')

  # if tf.test.gpu_device_name():
  #   print('Gpu device name:{}'  .format(tf.test.gpu_device_name()))
  # else:
  #   print('please install gpu')

  parser=argparse.ArgumentParser()
  
  ImageType = 'real' # real, imaginary
  modelName = 'Proposedmse'
  lmda = 0.01;
  task = 'train' # task = 'compress', 'decompress', or 'train'
  HolBaseName = 'Hol_3D_multi'  #Astronaut_Hol_v2, horse_Hol_v1_imag8B, Hol_3D_multi_imag8B
  Folder = 'Hol_3D_multi' #Hol_3D_multi
  lmdaStr = str(lmda);
  lmdaStr = lmdaStr.replace(".", "p")
  ModelNameFinal = modelName +"_" + lmdaStr
  name = ModelNameFinal+"_{}".format(int(time.time()))

  logdir ="logs/{}".format(name)

  parser.add_argument("--model_path", default = ModelNameFinal ,help="Path where to save/load the trained model.")
  parser.add_argument("--train_path", default=logdir,help="Path where to log training metrics for TensorBoard and back up ""intermediate model checkpoints.")
  parser.add_argument("--epochs", type=int, default=10,help="Train up to this number of epochs. (One epoch is here defined as ""the number of steps given by --steps_per_epoch, not iterations ""over the full training dataset.)")
  parser.add_argument("--steps_per_epoch", type=int, default=10,help="Perform validation and produce logs after this many batches.")
  parser.add_argument( "--max_validation_steps", type=int, default=16,help="Maximum number of batches to use for validation. If -1, use one " "patch from each image in the training set.")
  parser.add_argument("--num_scales", type=int, default=64,help="Number of Gaussian scales to prepare range coding tables for.")
  parser.add_argument("--scale_min", type=float, default=0.11,help="Minimum value of standard deviation of Gaussians.")
  parser.add_argument("--scale_max", type=float, default=256.,help="Maximum value of standard deviation of Gaussians.")
  parser.add_argument("--check_numerics", action="store_true",help="Enable TF support for catching NaN and Inf in tensors.")
  parser.add_argument('--verbose', '-V', action='store_true', help='Report bitrate and distortion when training or compressing.')
  parser.add_argument("--num_filters", type=int, default=192)


  #input_real_png = HolBaseName + '_real8B.png'
  #input_imag_png = HolBaseName + '_imag8B.png'
  #compressed_real = input_real_png[:-4] + "_" + ModelNameFinal + ".tfci"
  #compressed_imag = input_imag_png[:-4] + "_" + ModelNameFinal + ".tfci"
  #rec_real_png = input_real_png[:-4] + "_" + ModelNameFinal + "_rec.png"
  #rec_imag_png = input_imag_png[:-4] + "_" + ModelNameFinal + "_rec.png"
  #In_real_png_Path = os.path.join(Folder, input_real_png)
  #In_imag_png_Path = os.path.join(Folder, input_imag_png)
  #Cmp_real_path = os.path.join(Folder, compressed_real)
  #Cmp_imag_path = os.path.join(Folder, compressed_imag)
  #rec_real_path = os.path.join(Folder, rec_real_png)
  #rec_imag_path = os.path.join(Folder, rec_imag_png)
 #
  #if ImageType == 'real':
  #  input  = In_real_png_Path
  #  output = Cmp_real_path
  #  rec = rec_real_path
  #elif ImageType == 'imaginary':
  #  input = In_imag_png_Path
  #  output = Cmp_imag_path
  #  rec = rec_imag_path

  parser.add_argument('--command',type=str,default=task,help='')

  if task =='compress':
    parser.add_argument('--input_file',type=str,default= '1.png', help='input file name to compress')
    parser.add_argument('--output_file',type=str,default='1.tfci', help='output file name')
  elif task == 'decompress':
    parser.add_argument('--original',type=str,default='1.png', help='input file name to compress')
    parser.add_argument('--input_file',type=str,default='1.tfci', help='input file name to compress')
    parser.add_argument('--output_file',type=str,default='rec_1.png', help='output file name')

  parser.add_argument('--target_bpp',type=float,default=5,help='bpp')
  parser.add_argument('--bpp_strict',type=bool,default='False',help='')


  #for training
  parser.add_argument('--train_glob',type=str,default='training_set/*.png',help='directory of images for training.')
  parser.add_argument("--preprocess_threads", type=int, default=24,help="Number of CPU threads to use for parallel decoding of training images.")
  parser.add_argument("--batchsize", type=int, default=8,help="Batch size for training.")
  parser.add_argument("--patchsize", type=int, default=256,help="Size of image patches for training.")
  parser.add_argument("--lambda", type=float, default=lmda, dest="lmbda",help="Lambda for rate-distortion tradeoff.")

  args=parser.parse_args()
  print(args)
  main(args)
  #app.run(main, flags_parser=parse_args)