import cv2
import numpy as np


def nearest_interpolation(img, target_size: tuple):
    # 目标尺寸
    h, w = target_size

    # 原始尺寸
    if len(img.shape) > 2:
        height, width, channel = img.shape
    else:
        height, width = img.shape
        channel = 1

    # 缩放比率
    ratio_h = h / height
    ratio_w = w / width

    # 生成坐标矩阵
    y_index = np.arange(h, dtype=np.int32)
    x_index = np.arange(w, dtype=np.int32)
    y_target_index = np.int32(y_index / np.array(ratio_h))
    x_target_index = np.int32(x_index / np.array(ratio_w))

    # 根据像素个数指定原图坐标
    x_target_index = np.tile(x_target_index, h)
    y_target_index = y_target_index.repeat(w)

    img_new = img[y_target_index, x_target_index]
    img_new = img_new.reshape(h, w, channel)

    cv2.imshow("resized", img_new)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = "./pics/lenna.png"
    img_bgr = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    nearest_interpolation(img_gray, (680, 400))
