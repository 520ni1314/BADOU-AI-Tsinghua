import cv2
import numpy as np


def nearest_interpolation(img, target_size: tuple):
    height, width, channel = img.shape
    target_size = (target_size[0], target_size[1], channel)
    img_new = np.zeros(target_size, dtype=img.dtype)

    ratio_h = target_size[0] / height
    ratio_w = target_size[1] / width

    for i in range(0, target_size[0]):
        for j in range(0, target_size[1]):
            print()


if __name__ == '__main__':
    img_path = "./pics/lenna.png"
    img_bgr = cv2.imread(img_path)
    nearest_interpolation(img_bgr, [128, 128])
