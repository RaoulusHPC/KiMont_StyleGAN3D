import argparse
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import tensorflow as tf
import numpy as np

from models.stylegan import StyleGAN
from training import dataset


@dataclass
class ModelParameters:
    img_dim: int = 64
    latent_size: int = 160
    label_size: int = 9
    num_layers: int = 8
    gen_filters: List = field(default_factory=lambda:[128, 128, 64, 32, 16])
    disc_filters: List = field(default_factory=lambda:[16, 32, 64, 64, 128, 128, 128])
    # gen_filters: List = field(default_factory=lambda:[256, 128, 64, 32])
    # disc_filters: List = field(default_factory=lambda:[16, 32, 64, 128, 192, 224])



strategy = tf.distribute.get_strategy()
with strategy.scope():
    model_parameters = ModelParameters()
    model = StyleGAN(model_parameters=model_parameters)
    model.build()
    model.summary()

fake_latents = tf.random.normal(shape=(1, 512))
label = tf.constant([[0,0,1,0,0,0,0,0,0]])

style = model.generator.mapping_network(z=fake_latents,labels=label)
fake_images = model.generator(
                z=fake_latents,
                labels=label,
                training=False)

print("aa")



