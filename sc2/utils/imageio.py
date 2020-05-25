import numpy as np
import glob
import os
import sc2


class ImageioConfiguration(object):
    default_load_mode = "cv2"
    default_save_mode = "cv2"
    default_show_mode = "cv2"


config = ImageioConfiguration()


def configurate(name: str, value):
    if hasattr(config, name):
        setattr(config, name, value)
    else:
        raise ValueError(f"ImageioConfiguration has no attribute {name}")


def image_load(filename: str, mode=None) -> np.ndarray:
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)

    mode = config.default_load_mode if mode is None else mode

    if basename.find(".") == -1:
        # unspecified filetype, detect it by glob
        names = glob.glob(filename + ".*")
        
        if len(names) == 0:     # no file found
            if len(dirname) == 0:
                sc2.warnings.warn(f"Searching for image in images folder")
                return image_load(os.path.join("images", filename), mode=mode)
            raise ValueError(f"File not found: {filename}")
        else:
            if len(names) > 1:  # more than one file returned by glob
                sc2.warnings.warn(f"Found more than one: {names}, using {names[0]}")
            filename = names[0]
    
    if not os.path.isfile(filename):
        raise ValueError(f"File not exists: {filename}")

    if mode.lower() == "cv2":
        # load image using cv2
        from cv2 import imread
        from cv2 import cvtColor, COLOR_BGR2RGB
        image = imread(filename)
        # cv2 use BGR by default, transform to RGB for convenience
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cvtColor(image, COLOR_BGR2RGB)
        return image
    elif mode.lower() == "pil":
        # load image using PIL
        from PIL import Image
        image = Image.load(filename)
        image = np.asarray(image)
        return image
    else:
        raise ValueError(f"Only support cv2 and pil mode, but {mode} given")


def image_save(filename: str, image: np.ndarray, mode=None):
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)

    mode = config.default_save_mode if mode is None else mode

    # create folder
    if len(dirname) != 0:
        os.makedirs(dirname, exist_ok=True)
    
    # use png as default filetype
    if basename.find(".") == -1:
        basename = basename + ".png"

    image = np.copy(image)
    
    if mode.lower() == "cv2":
        # save image using cv2
        from cv2 import imwrite
        from cv2 import cvtColor, COLOR_RGB2BGR
        # convert RGB to BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cvtColor(image, COLOR_RGB2BGR)
        imwrite(filename, image)
    elif mode.lower() == "pil":
        # read image using PIL
        from PIL import Image
        image = Image.fromarray(image)
        image.save(filename)
    else:
        raise ValueError(f"Only support cv2 and pil mode, but {mode} given")


def image_show(image: np.ndarray, name="example", mode=None, freeze=False):
    image = np.copy(image)

    mode = config.default_show_mode if mode is None else mode

    if mode.lower() == "cv2":
        from cv2 import namedWindow, imshow, waitKey
        from cv2 import cvtColor, COLOR_RGB2BGR
        namedWindow(name)
        # convert RGB to BGR
        image = cvtColor(image, COLOR_RGB2BGR)
        imshow(name, image)
        if freeze:
            return waitKey()
        else:
            return waitKey(1)
    elif mode.lower() == "plt":
        from matplotlib import pyplot as plt
        plt.imshow(image)
        plt.show()
    else:
        raise ValueError(f"Only support cv2 and plt mode, but {mode} given")
