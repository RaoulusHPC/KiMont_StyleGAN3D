
import tensorflow as tf
from tensorflow.keras import losses
from dataclasses import dataclass


#from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers, losses
import numpy as np
from pathlib import Path
from random import *
from training import dataset
from models.base_models import Comparator

class TrainingArguments:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    adam_eps: float = 1e-7
    val_size: int = 200


parameters = TrainingArguments()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if len(gpus) > 2:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

tfrecords = ['data/simpleGRAB_1000.tfrecords']
tf_dataset = dataset.get_simplegrab_dataset(tfrecords)

val_dataset = tf_dataset.take(parameters.val_size).shuffle(2048, reshuffle_each_iteration=True).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = strategy.experimental_distribute_dataset(val_dataset)
train_dataset = tf_dataset.skip(parameters.val_size).shuffle(2048, reshuffle_each_iteration=True).map(lambda x, y: dataset.simple_grab_aug(x, y)).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)
train_dataset = strategy.experimental_distribute_dataset(train_dataset)

checkpoint_dir = 'ckpts/comparator/'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        save_freq='epoch',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs/comparator'
)

with strategy.scope():
    comparator = Comparator()
    comparator_input = tf.zeros(shape=(1, 64, 64, 64, 1))
    comparator((comparator_input, comparator_input))
    comparator.summary()
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        1e-3, 1000/parameters.batch_size * parameters.epochs, alpha=0.0, name=None
    )
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=parameters.adam_eps)
    comparator.compile(optimizer=opt, loss=losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

comparator.fit(
    train_dataset,
    validation_data=val_dataset,
    verbose=1,
    epochs=parameters.epochs,
    callbacks=[checkpoint_callback, tensorboard_callback])

comparator.load_weights(checkpoint_dir)
comparator.evaluate(val_dataset)

