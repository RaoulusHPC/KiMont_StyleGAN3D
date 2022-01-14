
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
import scipy
import pickle
import math
import os
from tqdm import tqdm
from training import dataset
from training.visualize import Screenshotter

# https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI

def calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake):
    m = np.square(mu_fake - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
    dist = m + np.trace(sigma_fake + sigma_real - 2*s)
    return np.real(dist)


def calculate_slice_fid(generator, num_fakes, real_dataset, batch_size):

        img_dim = generator.img_dim
        upscale_factor = int(math.ceil(75 / img_dim))
        upscaled_img_dim = upscale_factor * img_dim

        inception_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(upscaled_img_dim, upscaled_img_dim, 3),
            pooling='avg')

        def calculate_statistics(dataset):
            feat = inception_model.predict(dataset, verbose=1)
            mu = np.mean(feat, axis=0)
            sigma = np.cov(feat, rowvar=False)
            return mu, sigma

        cache_file = './cache/slice_fid.pickle'
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        # Define real datasets
        upscaler = tf.keras.layers.UpSampling2D(size=upscale_factor)

        label_dataset =  real_dataset.map(lambda x, y: y)
        real_dataset = real_dataset.map(lambda x, y: tf.tile(x, [1, 1, 1, 1, 3]))
        real_dataset = real_dataset.map(lambda x: 0.5 * (x + 1))

        sag_real_dataset = real_dataset.map(lambda x: x[:, img_dim // 2])
        sag_real_dataset = sag_real_dataset.map(lambda x: upscaler(x))

        axi_real_dataset = real_dataset.map(lambda x: x[:, :, img_dim // 2])
        axi_real_dataset = axi_real_dataset.map(lambda x: upscaler(x))

        cor_real_dataset = real_dataset.map(lambda x: x[:, :, :, img_dim // 2])
        cor_real_dataset = cor_real_dataset.map(lambda x: upscaler(x))

        if os.path.isfile(cache_file):
            print('Loading real statistics')
            with open(cache_file, 'rb') as f:
                sag_mu_real, sag_sigma_real, axi_mu_real, axi_sigma_real, cor_mu_real, cor_sigma_real = pickle.load(f)   
        else:
            print('Calculating real statistics')
            sag_mu_real, sag_sigma_real = calculate_statistics(sag_real_dataset)
            axi_mu_real, axi_sigma_real = calculate_statistics(axi_real_dataset)
            cor_mu_real, cor_sigma_real = calculate_statistics(cor_real_dataset)

            with open(cache_file, 'wb') as f:
                pickle.dump((sag_mu_real, sag_sigma_real, axi_mu_real, axi_sigma_real, cor_mu_real, cor_sigma_real), f)

        # Generate fake datasets
        with tf.device('/CPU:0'):
            sag_fake_dataset = []
            axi_fake_dataset = []
            cor_fake_dataset = []
        print('Generating fake data')

        label_iterator = iter(label_dataset)
        for _ in tqdm(range(0, num_fakes, batch_size)):
            fake_latents = tf.random.normal(shape=(batch_size, generator.latent_size))
            # labels = dataset.get_random_labels(batch_size=batch_size, label_size=generator.label_size)
            labels = next(label_iterator)
            generated_images = generator(
                z=fake_latents,
                labels=labels,
                training=False)

            # add slices to fake datasets
            sag_fake_dataset.append(generated_images[:, generator.img_dim // 2])
            axi_fake_dataset.append(generated_images[:, :, generator.img_dim // 2])
            cor_fake_dataset.append(generated_images[:, :, :, generator.img_dim // 2])

        def prepare_fake_dataset(fake_dataset):
            fake_dataset = tf.concat(fake_dataset, axis=0)[:num_fakes]
            fake_dataset = tf.where(fake_dataset > 0, 1., 0.)
            fake_dataset = upscaler(fake_dataset)
            fake_dataset = tf.tile(fake_dataset, [1, 1, 1, 3])
            return fake_dataset

        with tf.device('/CPU:0'):
            sag_fake_dataset = prepare_fake_dataset(sag_fake_dataset)
            axi_fake_dataset = prepare_fake_dataset(axi_fake_dataset)
            cor_fake_dataset = prepare_fake_dataset(cor_fake_dataset)

        print('Calculating fake statistics')
        sag_mu_fake, sag_sigma_fake = calculate_statistics(sag_fake_dataset)
        axi_mu_fake, axi_sigma_fake = calculate_statistics(axi_fake_dataset)
        cor_mu_fake, cor_sigma_fake = calculate_statistics(cor_fake_dataset)

        sag_fid = calculate_fid(sag_mu_real, sag_sigma_real, sag_mu_fake, sag_sigma_fake)
        axi_fid = calculate_fid(axi_mu_real, axi_sigma_real, axi_mu_fake, axi_sigma_fake)
        cor_fid = calculate_fid(cor_mu_real, cor_sigma_real, cor_mu_fake, cor_sigma_fake)
        
        return sag_fid, axi_fid, cor_fid


def calculate_screenshot_fid(mapping_network, generator, num_fakes, real_dataset, batch_size):

        screenshot_resolution = 128
        screenshotter = Screenshotter(screenshot_resolution=screenshot_resolution)

        inception_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(screenshot_resolution, screenshot_resolution, 3),
            pooling='avg')

        def calculate_statistics(dataset):
            feat = inception_model.predict(dataset, verbose=1)
            mu = np.mean(feat, axis=0)
            sigma = np.cov(feat, rowvar=False)
            return mu, sigma

        cache_file = './cache/screenshot_fid.pickle'
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        if os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                mu_real, sigma_real = pickle.load(f)   
        else:
            real_screenshots = []
            print('Taking screenshots of real data')
            for data, _ in tqdm(real_dataset):
                for component in list(data):
                    screenshot = screenshotter(component)
                    real_screenshots.append(screenshot)

            real_screenshots = tf.stack(real_screenshots)
            mu_real, sigma_real = calculate_statistics(real_screenshots)

            with open(cache_file, 'wb') as f:
                pickle.dump((mu_real, sigma_real), f)

        # Generate fake datasets
        fake_screenshots = []

        for _ in tqdm(range(0, num_fakes, batch_size)):
            fake_latents = tf.random.normal(shape=(batch_size, mapping_network.latent_size))
            labels = dataset.get_random_labels(batch_size=batch_size, label_size=mapping_network.label_size)
            w = mapping_network(
                z=fake_latents,
                labels=labels,
                training=False)
            w = tf.tile(tf.expand_dims(w, axis=1), [1, generator.num_ws, 1])
            generated_images = generator(
                w=w,
                training=False)

            binarized_images = tf.where(generated_images > 0, 1., 0.)
            for binarized_image in binarized_images:
                screenshot = screenshotter(binarized_image)
                fake_screenshots.append(screenshot)

        fake_screenshots = tf.stack(fake_screenshots)
        mu_fake, sigma_fake = calculate_statistics(fake_screenshots)

        fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        
        return fid



        
