import numpy as np


def draw_mask(image: np.ndarray) -> np.ndarray:
    from cv2 import (
        imshow,
        circle,
        waitKey,
        cvtColor,
        namedWindow, 
        destroyWindow,
        setMouseCallback, 
        COLOR_RGB2BGR,
        EVENT_FLAG_LBUTTON
    )

    image = np.copy(image)
    image = cvtColor(image, COLOR_RGB2BGR)
    result = np.zeros((image.shape[0], image.shape[1]), dtype=np.int8)

    def draw_circle(event, x, y, flags, param):
        if flags == EVENT_FLAG_LBUTTON:
            circle(image, (x, y), 2, (0, 0, 255), -1)
            circle(result, (x, y), 2, 1, -1)

    name = "mask (press Q to quit)"
    namedWindow(name)
    setMouseCallback(name, draw_circle)
    
    while True:
        imshow(name, image)
        if waitKey(1) == ord('q'):
            break
    
    destroyWindow(name)
    
    return result
