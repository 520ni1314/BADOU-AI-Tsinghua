
##################################
# 双线性插值实现
##################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

def bilinear(img,out_dim):
    # 读取图像尺寸h,w,c
    src_h, src_w, c = img.shape
    dst_h, dst_w = out_dim[0],out_dim[1]
    # 创建全空图像emptyImage,out_dim*out_dim
    dst_img = np.zeros((dst_h,dst_w,c), np.uint8)
    scale_h = float(src_h)/dst_h
    scale_w = float(src_w)/dst_w

    ##################
    # for循环
    ##################
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 中心点归一化
                src_x = (dst_x + 0.5) * scale_h - 0.5
                src_y = (dst_y + 0.5) * scale_w - 0.5
                # 找到将用于计算插值的点的坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 进行第二次插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

if __name__ == '__main__':
    # 读取图像
    img = cv2.imread('lenna.png')
    # 双线性插值操作
    img_bilinear = bilinear(img,(700,700))
    cv2.imshow("image_bilinear",img_bilinear)
    cv2.waitKey(0)
    plt.imsave('lenna_bilinear.jpg',img_bilinear)