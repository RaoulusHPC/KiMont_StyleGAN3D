
import os
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from training import dataset, visualize
from optimization.optimizer import LatentOptimizer
from models.base_models import Comparator, LatentMapper
from models.stylegan import StyleGAN
from train_stylegan import ModelParameters

if __name__ == "__main__":

    train_dataset = dataset.get_projected_dataset('data/projected_images.tfrecords')
    train_dataset = train_dataset.batch(1)
    train_dataset = train_dataset.skip(5000)
    train_dataset.shuffle(2000)

    # for d in train_dataset:
    #     original_image, generated_image, label, w = d
    #     if tf.math.reduce_mean(tf.math.square(original_image - generated_image)) < 0.005:
    #         break

    model = StyleGAN(model_parameters=ModelParameters())
    model.build()
    model.setup_moving_average()
    ckpt = tf.train.Checkpoint(
        seen_images=tf.Variable(0),
        generator_ema=model.generator_ema)
    manager = tf.train.CheckpointManager(
        ckpt,
        directory='ckpts/stylegan/',
        max_to_keep=None)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    generator = model.generator_ema

    comparator = Comparator()
    comparator_input = tf.zeros(shape=(1, 64, 64, 64, 1))
    comparator((comparator_input, comparator_input))
    comparator.load_weights('ckpts/comparator/')

    mapper_grab = LatentMapper(generator.latent_size, num_layers=4)
    mapper_grab.build((None, generator.latent_size))
    mapper_grab.load_weights('ckpts/mapper/mapper_2.0_0.5_0.5/20220128-132753/')

    mapper_grab0 = LatentMapper(generator.latent_size, num_layers=4)
    mapper_grab0.build((None, generator.latent_size))
    mapper_grab0.load_weights('ckpts/mapper/mapper_2.0_1.0_0.0/20220128-120210/')

    mapper_grab1 = LatentMapper(generator.latent_size, num_layers=4)
    mapper_grab1.build((None, generator.latent_size))
    mapper_grab1.load_weights('ckpts/mapper/mapper_2.0_0.0_1.0/20220128-123600/')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = 'results/optimization/' + current_time
    os.makedirs(results_dir, exist_ok=True)

    def binarize(x):
        return tf.where(x > 0, 1.0, -1.0)

    c = 0
    for original_image, generated_image, label, w  in train_dataset:

        save_dir = os.path.join(results_dir, str(c))
        os.mkdir(save_dir)

        plots = [original_image]
        visualize.screenshot_and_save([original_image], filepath=os.path.join(save_dir, 'original.png'), window_size=(1000, 1000))

        optimizer = LatentOptimizer(generator, comparator, steps=200, lambda0=1., lambda1=0.)
        optimized_image, _, _ = optimizer.optimize(w)
        plots.append(optimized_image)
        visualize.screenshot_and_save([optimized_image], filepath=os.path.join(save_dir, 'grab0.png'), window_size=(1000, 1000))
        optimizer = LatentOptimizer(generator, comparator, steps=200, lambda0=0., lambda1=1.)
        optimized_image, _, _ = optimizer.optimize(w)
        plots.append(optimized_image)
        visualize.screenshot_and_save([optimized_image], filepath=os.path.join(save_dir, 'grab1.png'), window_size=(1000, 1000))
        optimizer = LatentOptimizer(generator, comparator, steps=200, lambda0=0.5, lambda1=0.5)
        optimized_image, _, _ = optimizer.optimize(w)
        plots.append(optimized_image)
        visualize.screenshot_and_save([optimized_image], filepath=os.path.join(save_dir, 'grab.png'), window_size=(1000, 1000))

        # optimized_image = binarize(generator.synthesize(w + mapper_grab0(w)))
        # plots.append(optimized_image)
        # visualize.screenshot_and_save([optimized_image], filepath=os.path.join(save_dir, 'mapper_grab0.png'), window_size=(1000, 1000))
        # optimized_image = binarize(generator.synthesize(w + mapper_grab1(w)))
        # plots.append(optimized_image)
        # visualize.screenshot_and_save([optimized_image], filepath=os.path.join(save_dir, 'mapper_grab1.png'), window_size=(1000, 1000))
        # optimized_image = binarize(generator.synthesize(w + mapper_grab(w)))
        # plots.append(optimized_image)
        # visualize.screenshot_and_save([optimized_image], filepath=os.path.join(save_dir, 'mapper_grab.png'), window_size=(1000, 1000))

        # filepath = f'{results_dir}/{c}.png'
        visualize.screenshot_and_save(plots[::-1], filepath=os.path.join(save_dir, 'opt_grab0_grab1_grab.png'), shape=(1, len(plots)), window_size=(len(plots) * 1000, 1000))

        # optimized_image = generator.synthesize(w + mapper_grab(w))
        # visualize.screenshot_and_save([optimized_image], filepath=f'{dir}/map_grab.png', window_size=(1000, 1000))
        # optimized_image = generator.synthesize(w + mapper_grab0(w))
        # visualize.screenshot_and_save([optimized_image], filepath=f'{dir}/map_grab0.png', window_size=(1000, 1000))
        # optimized_image = generator.synthesize(w + mapper_grab1(w))
        # visualize.screenshot_and_save([optimized_image], filepath=f'{dir}/map_grab1.png', window_size=(1000, 1000))

        # changes = tf.where((optimized_image - original_image) > 0, 1., -1.)
        # visualize.screenshot_and_save([original_image, generated_image, optimized_image, changes], filepath=f'{dir}mapper.png', shape=(1, 3), window_size=(3000, 1000))

        c += 1
        if c == 100:
            break