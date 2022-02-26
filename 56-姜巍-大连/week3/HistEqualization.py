import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Equalization:
    """直方图均衡化&输出均衡化后图像"""

    def __init__(self, src_img):
        """初始化创建表格用于存储计算数据"""
        self.df = pd.DataFrame(np.zeros((256, 4), dtype=np.int64), columns=['origin_pixel', 'sum', 'q', 'corr'])
        self.img = src_img
        self.h = src_img.shape[0]
        self.w = src_img.shape[1]

    def origin_pix(self):
        """第一列保存原始图像各灰度级数量"""
        for h in range(self.h):
            for w in range(self.w):
                self.df.iloc[self.img[h][w]]['origin_pixel'] += 1

    def sum_pix(self):
        """从0~当前灰度级的像素数量逐级累加求和"""
        self.df.iloc[0]['sum'] = self.df.iloc[0]['origin_pixel']
        for m in range(1, 256):
            self.df.iloc[m]['sum'] = self.df.iloc[m - 1]['sum'] + self.df.iloc[m]['origin_pixel']

    def cal_q(self):
        """按照定义式求均衡化，并将新的灰度级的数字对应存入元灰度级对应的label中"""
        for n in range(256):
            self.df.iloc[n]['q'] = np.int64(self.df.iloc[n]['sum'] * (256 / (self.h * self.w)) - 1)

    def correction(self):
        """将'cal_q'计算结果中的-1全部修正到0或正整数，同时保证各灰度级之间相对亮度近似关系不变"""
        # k = 0  # DataFrame的index，也可看做是原始灰度级的数值
        s = 0  # 存储需要修正的灰度级的个数
        temp_list = []  # 存储需要修正的灰度级所在index值
        for k in range(256):  # 从原始灰度级k = 0开始向后迭代，找到是否存在第一个-1值
            if self.df.iloc[k]['origin_pixel'] == 0:
                continue
            else:
                if self.df.iloc[k]['q'] != -1:  # 判断第一个值不为-1，则结束循环，跳转到输出图像步骤
                    print("无需修正！")
                    break
                else:  # 判断出现-1，要记录有多少个原始灰度级变换后为-1以及原始灰度级的值用于修正
                    while s >= self.df.iloc[k]['q']:
                        s += 1
                        temp_list.append(k)
        corr = 0  # 将需要修正的灰度级从0开始依次赋予新的灰度值，效果上所有的-1变成了0123...而计算结果中的0123依次向后推，直到不需要修正。
        for t in range(256):
            if t in temp_list:
                self.df.iloc[t]['corr'] = corr
                corr += 1
            else:
                self.df.iloc[t]['corr'] = self.df.iloc[t]['q']

    def dst_img(self):
        """输出均衡化后图像"""
        image = np.zeros((self.h, self.w), dtype=np.uint8)
        for i in range(self.h):
            for j in range(self.w):
                image[i][j] = self.df.iloc[self.img[i][j]]['corr']
        return image

    # for record: 该算法在计算时仍会出现一个已知的问题，即由于存储的数据类型为int64，
    # 导致某些连续的原始灰度级（该灰度级中的像素点数量所占总数的比例是很低的），在计算值后存储时会出现值相同的情况。
    # 后续待优化。


# read the 'lenna.png' cover to gary and RGB.
img_src = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_origin = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

# show the origin image of 'lenna'
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('lenna_gray', fontsize='small')

# show the grayscale histogram of 'lenna'
plt.subplot(2, 2, 2)
# 'np.ravel()' or 'np.flatten()' function aims to cover the input array to a 1-D array,
#   containing the elements of the input.
plt.hist(gray.flatten(), 256)
plt.title('grayscale histogram of \'lenna_gray\'', fontsize='small')

eg = Equalization(gray)  # 将均衡化类实例化
eg.origin_pix()  # 载入原始图像信息
eg.sum_pix()  # 逐级将灰度级个数累计求和
eg.cal_q()  # 根据定义计算得到均衡化后灰度级的初始值
eg.correction()  # 将-1灰度级进行修正
dst_image = eg.dst_img()  # 输出均衡化后灰度图像

plt.subplot(2, 2, 3)
plt.imshow(dst_image, cmap='gray')
plt.title('lenna_gray_qual', fontsize='small')

plt.subplot(2, 2, 4)
plt.hist(dst_image.flatten(), 256)
plt.title('grayscale histogram equalization of \'lenna_gray\'', fontsize='small')

plt.tight_layout()
plt.show()
