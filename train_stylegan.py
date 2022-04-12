
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


@dataclass
class TrainingArguments:
    train_kimg: int = 15_000
    batch_size_per_replica: int = 16
    global_batch_size: int = batch_size_per_replica

    gen_lr: float = 2e-3
    disc_lr: float = 1.5e-3
    adam_beta_1: float = 0.0
    adam_beta_2: float = 0.99
    adam_gen_eps: float = 1e-7

    r1_interval: int = 1
    r1_gamma: int = 10

    ema_kimg: int = 20
    ema_beta: int = None

    apa: bool = True
    apa_interval: int = 4
    tune_target: float = 0.6
    tune_kimg: int = 800

    print_interval: int = 1000
    evaluate_interval: int = 4000
    log_interval: int = 1000
    #metric_interval: int = 5000
    checkpoint_interval: int = 4000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", type=bool, default=True, help="use GPU")
    parser.add_argument("--memory_growth", type=bool, default=False, help="use memory growth")
    # parser.add_argument("--data_dir", type=str, default="/mnt/md0/Pycharm_Raid/datasets/mcb/32", help="path to folder containing data")

    args = parser.parse_args()
    os.environ["NCCL_DEBUG"] = "INFO"

    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')

    # if gpus:
    #     tf.config.set_logical_device_configuration(
    #     gpus[0],
    #     [tf.config.LogicalDeviceConfiguration(memory_limit=8192),
    #      tf.config.LogicalDeviceConfiguration(memory_limit=8192)])

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                if args.memory_growth:
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

    training_args = TrainingArguments()
    training_args.global_batch_size = training_args.batch_size_per_replica * strategy.num_replicas_in_sync
    training_args.ema_beta = 0.5 ** (training_args.global_batch_size / max(training_args.ema_kimg * 1000, 1e-8))

    model_parameters = ModelParameters()

    dataset_name = 'ABC'
    if dataset_name == 'MCB':
        tfrecords = ['data/mcb64_screws.tfrecords']
        tf_dataset = dataset.get_mcb_base(tfrecords)
        model_parameters.label_size = 9
    elif dataset_name == 'ABC':
        tfrecords = list(Path('data/abc/').rglob('*.tfrecords'))
        tf_dataset = dataset.get_abc_base(tfrecords)

    train_dataset = tf_dataset.shuffle(2048, reshuffle_each_iteration=True).repeat(5000).batch(training_args.global_batch_size).prefetch(tf.data.AUTOTUNE)
    fakelabel_dataset = tf_dataset.map(lambda _, y: y).shuffle(2048, reshuffle_each_iteration=True).repeat(5000).batch(training_args.global_batch_size).prefetch(tf.data.AUTOTUNE)
    
    dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_fakelabel_dataset = strategy.experimental_distribute_dataset(fakelabel_dataset)
    dist_train_dataset_iterator = iter(dist_train_dataset)
    dist_fakelabel_dataset_iterator = iter(dist_fakelabel_dataset)

    with strategy.scope():
        model = StyleGAN(model_parameters=model_parameters)
        model.build()
        model.summary()
        model.compile(training_args=training_args)
    
    @tf.function
    def distributed_train_step(dist_data, dist_real_labels, dist_fake_labels):
        strategy.run(model.train_step, args=(dist_data, dist_real_labels, dist_fake_labels))

    while int(model.ckpt.seen_images.read_value()) // 1000 < training_args.train_kimg:

        dist_data, dist_real_labels = next(dist_train_dataset_iterator)
        dist_fake_labels = next(dist_fakelabel_dataset_iterator)

        distributed_train_step(dist_data, dist_real_labels, dist_fake_labels)

        model.update_moving_average()

        model.ckpt.step.assign_add(1)
        model.ckpt.seen_images.assign_add(model.args.global_batch_size)

        step = int(model.ckpt.step.read_value())

        if step % model.args.checkpoint_interval == 0:
            model.checkpoint()

        if step % model.args.evaluate_interval == 0:
            model.evaluate()

        if step % model.args.print_interval == 0:
            model.print_info()
        
        if step % model.args.log_interval == 0:
            model.log_metrics()

        if step % model.args.apa_interval == 0:
            # rt = (model.training_metrics['loss_signs_real'] + model.training_metrics['loss_signs_fake']) / 2
            rt = model.training_metrics['loss_signs_real'].result()
            nimg_delta = model.args.apa_interval * model.args.global_batch_size
            nimg_ratio = nimg_delta / (model.args.tune_kimg * 1000)
            deception_strength = model.ckpt.deception_strength.read_value() + nimg_ratio * np.sign(rt - model.args.tune_target)
            deception_strength = min(max(deception_strength, 0), 0.9)

            model.ckpt.deception_strength.assign(deception_strength)

            model.training_metrics['loss_signs_real'].reset_states()
            model.training_metrics['loss_signs_fake'].reset_states()

        if step % model.args.log_interval == 0:
            for _, metric in model.training_metrics.items():
                metric.reset_states()

if __name__ == "__main__":
    main()