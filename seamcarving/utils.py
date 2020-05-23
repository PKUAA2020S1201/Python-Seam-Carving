import numpy as np
import numba as nb
import cv2
from PIL import Image
from warnings import simplefilter
from matplotlib import pyplot as plt


def normalized(energy):
    return energy / np.max(energy)


def ignore_numba_warnings():
    simplefilter("ignore", nb.NumbaWarning)


def load_image(filename):
    return cv2.imread(filename)


def save_image(filename, image):
    cv2.imwrite(filename, image)


def show_image_cv2(image, name="seamcarving", freeze=False):
    cv2.imshow(name, image)
    if freeze:
        cv2.waitKey()
    else:
        cv2.waitKey(1)


def show_image_plt(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def image_transpose(image):
    if len(image.shape) == 2:
        return np.transpose(image, [1, 0])
    elif len(image.shape) == 3:
        return np.transpose(image, [1, 0, 2])
    else:
        raise NotImplementedError(
            f"image_transpose not implemented for image with shape {image.shape}"
        )
