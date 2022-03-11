#####################
# dHash算法
"""
    算法步骤：
    step1： 缩放
    step2： 灰度化
    step3： 比较
    step4： 生成Hash
    step5： 对比指纹，计算汉明距离
"""
#####################

import cv2
import numpy as np

def aHash(img):
    # step1：缩放
    img = cv2.resize(img, (9,8))
    # step2：灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 设置Hash值初始值Hash_str为''
    Hash_str = ''
    # step3:比较
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > img_gray[i,j+1]:
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
    for i in range(len(Hash1)):
        if Hash1[i] != Hash2[i]:
            n = n + 1
    return n

# 差值哈希算法
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
print("插值哈希算法的相似度：",n)