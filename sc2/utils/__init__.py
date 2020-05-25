from . import jit
from . import mask
from . import imageio
from . import normalization


from .imageio import image_load, image_save, image_show
# alias for image_load
image_read = imread = imload = image_load
# alias for image_save
image_write = imwrite = imsave = image_save
# alias for image_show
display = imshow = image_show


from .mask import draw_mask
# alias for draw_mask
read_mask = get_mask = draw_mask


from .jit import just_in_time
# from numba import jit as just_in_time


from .transpose import transpose


from .normalization import (
    sigmoid_normalization,
    tanh_normalization,
    max_abs_normalization,
    min_max_normalization,
    mean_std_normalization
)
