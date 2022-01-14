
import tensorflow as tf
import numpy as np
import pyvista as pv
import matplotlib
import os
from math import sqrt, ceil, floor

if os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False):
    pv.start_xvfb()

def visualize_npy_file(file):
    tensor = np.load(file)
    visualize_tensors(tensor)


def visualize_tensors(tensors, shape: tuple=None, shade: bool=True):
    if not shape:
        dim = sqrt(len(tensors))
        shape = (ceil(dim), floor(dim)) # As close to a quadrilateral as possible
    plotter = pv.Plotter(shape=shape)
    plotter.set_background('white')
    _add_tensors_to_plotter(plotter, tensors, shape, shade)
    plotter.show()


def rescale_tensor(tensor, new_max=1, new_min=0):
    min, max = np.min(tensor), np.max(tensor)
    tensor = (tensor-min)/(max-min)
    tensor = tensor*(new_max-new_min) + new_min
    return tensor


def screenshot_and_save(tensors, filepath: str, shape: tuple=None, window_size: tuple=None):
    if not shape:
        dim = sqrt(len(tensors))
        shape = (ceil(dim), floor(dim))
    if not window_size:
        window_size = (len(tensors) * 128, len(tensors) * 128)
    plotter = pv.Plotter(shape=shape, off_screen=True)
    plotter.set_background('white')
    _add_tensors_to_plotter(plotter, tensors, shape)
    plotter.screenshot(filename=filepath, window_size=window_size)
    plotter.deep_clean()
    plotter.close()
    del plotter

def _add_tensors_to_plotter(plotter: pv.Plotter, tensors, shape: tuple, shade: bool=True):
    for i in range(shape[0]):
        for j in range(shape[1]):
            if tensors:
                tensor = tensors.pop()
                tensor = np.squeeze(np.array(tensor))
                plotter.subplot(i, j)
                plotter.add_volume(tensor, cmap="viridis", shade=shade)


class Screenshotter:
    def __init__(self, screenshot_resolution=128):
        self.screenshot_resolution = screenshot_resolution
        self.screenshot_cmap = matplotlib.colors.ListedColormap(['black', 'white'])
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background('black')

    def __call__(self, volume, filename=None):
        self.plotter.add_volume(np.squeeze(volume), cmap=self.screenshot_cmap, show_scalar_bar=False)
        screenshot = self.plotter.screenshot(filename=filename, window_size=[self.screenshot_resolution, self.screenshot_resolution])
        self.plotter.clear()
        return screenshot

if __name__ == '__main__':
    x = tf.ones((1, 32, 32, 32, 1))
    x = tf.keras.layers.ZeroPadding3D(padding=(16, 16, 16))(x)
    x = 2 * x - 1
    x = -x
    import matplotlib
    import matplotlib.pyplot as plt
    resolution = 128
    pv.set_plot_theme("dark")
    cmap = matplotlib.colors.ListedColormap(['black', 'white'])
    plotter = pv.Plotter(off_screen=True)
    plotter.add_volume(np.squeeze(x), cmap=cmap, show_scalar_bar=False)
    screenshot = plotter.screenshot('test.png', window_size=[resolution, resolution])
    print(screenshot.shape)
    plt.imshow(screenshot)
    print(1)
    # visualize_tensors([x])