#####################
# aHash算法
"""
    算法步骤：
    step1： 缩放
    step2： 灰度化
    step3： 求平均值
    step4： 比较
    step5： 生成Hash
    step6： 对比指纹，计算汉明距离
"""
#####################

import cv2
import numpy as np

def aHash(img):
    # step1：缩放
    img = cv2.resize(img, (8,8))
    # step2：灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 设置像素和初始值s为0，Hash值初始值Hash_str为''
    s = 0
    Hash_str = ''
    # 遍历累加所有像素和8*8
    for i in range(8):
        for j in range(8):
            s = s + img_gray[i,j]
    # step3：求平均值
    avg = s / 64
    # step4:比较
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > avg:
                Hash_str = Hash_str + '1'
            else:
                Hash_str = Hash_str + '0'
    # step5:生成Hash
    return Hash_str

# step6：对比指纹，计算汉明距离
def cmpHash(Hash1,Hash2):
    # 设置初始汉明距离n为0
    n = 0
    # 首先判断两个Hash长度是否相同
    if len(Hash1) != len(Hash2):
        return -1
    # 计算汉明距离
    for i in range(len(Hash2)):
        if Hash1[i] != Hash2[i]:
            n = n + 1
    return n

# 均值哈希算法
# 读取图片
img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
# 计算图片对应的Hash值
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
# 对比，计算汉明距离
n = cmpHash(hash1,hash2)
print("均值哈希算法的相似度：",n)