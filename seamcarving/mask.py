import numpy as np
import numba as nb
import cv2


def draw_mask(image):
    image = np.copy(image)
    result = np.zeros((image.shape[0], image.shape[1]))
    def draw_circle(event, x, y, flags, param):
        if flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            cv2.circle(result, (x, y), 2, 1, -1)
    cv2.namedWindow("mask")
    cv2.setMouseCallback("mask", draw_circle)
    while True:
        cv2.imshow("mask", image)
        if cv2.waitKey(1) == ord('q'):
            break
    return result


def energy_with_remove_mask(energy, mask, constant=-1000.0):
    energy = np.copy(energy)
    energy[mask != 0] = constant
    return energy


def energy_with_protect_mask(energy, mask, constant=np.inf):
    energy = np.copy(energy)
    energy[mask != 0] = constant
    return energy


if __name__ == "__main__":
    print(cv2.__version__)
    image = cv2.imread("images/jinge.jpg")
    mask = draw_mask(image)
