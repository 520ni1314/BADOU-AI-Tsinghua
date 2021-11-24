# import numpy and opencv

import numpy as np
import cv2

# write bilinear interpolation function
# input image and dimension

def bilinear_interpolation(img,dimensions):
    # unpack and output dimensions
    src_h, src_w, channel = img.shape
    dst_w, dst_h = dimensions
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # calculate scale
    scale_y, scale_x = float(src_h)/dst_h, float(src_w)/dst_w

    # create matrix for new image
    new_image = np.zeros((dst_h,dst_w,3), dtype=np.uint8)
    # loop through each pixel and calculate the value
    for dst_y in range(dst_h):
        for dst_x in range(dst_x):
            src_x = (dst_x + 0.5) * (scale_x) - 0.5
            src_y = (dst_y + 0.5) * (scale_y) - 0.5
            # get Q11, Q12, Q21, Q22 cooradinate
            src_x1 = int(src_x)
            src_x2 = min(src_1 + 1, src_w - 1)
            src_y1 = int(src_y)
            src_y2 = min(src_1 + 1, src_h - 1)

            f1 = (src_2 - src_x) *
