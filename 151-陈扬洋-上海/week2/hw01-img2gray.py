import cv2
import numpy as np


def img2gray(img, mode=0):
    img_array = np.array(img)

    height, weight, channel = img_array.shape
    img_gray_array = np.zeros([height, weight], dtype=img.dtype)
    img_b = img_array[:, :, 0]
    img_g = img_array[:, :, 1]
    img_r = img_array[:, :, 2]

    if mode == 1:
        for h in range(0, height):
            for w in range(0, weight):
                img_gray_array[h, w] = np.uint8((76 * img_r[h, w] + 151 * img_g[h, w] + 28 * img_b[h, w]) >> 8)

    elif mode == 2:
        for h in range(0, height):
            for w in range(0, weight):
                img_gray_array[h, w] = np.uint8((30 * img_r[h, w] + 59 * img_g[h, w] + 11 * img_b[h, w]) / 100)

    else:
        for h in range(0, height):
            for w in range(0, weight):
                img_gray_array[h, w] = np.uint8(0.3 * img_r[h, w] + 0.59 * img_g[h, w] + 0.11 * img_b[h, w])

    cv2.imshow("gray", img_gray_array)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = "./pics/lenna.png"
    img_bgr = cv2.imread(img_path)
    img2gray(img_bgr, 2)
