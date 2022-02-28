import cv2
import numpy as ny
import cv2 as cv

def average_Hash(src_img,w,h):
    s=0
    str_hash = ""
    deal_img=cv.imread(src_img)
    new_img=cv.resize(deal_img,(w,h))
    gray_img=cv.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    for i in range(w):
        for j in range(h):
            s=s+gray_img[i,j]
    avg=s/(w*h)
    for i in range(w):
        for j in range(h):
            if gray_img[i,j]>=avg:
                str_hash=str_hash+"1"
            if gray_img[i,j]<avg:
                str_hash=str_hash+"0"
    return(str_hash)

def deviation_Hash(src_img,w,h):
    str_hash=""
    deal_img=cv.imread(src_img)
    new_img=cv.resize(deal_img,(w,h))
    gray_img=cv.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    for i in range(w-1):
        for j in range(h):
            if gray_img[i,j]>gray_img[i,j+1]:
                str_hash=str_hash+"1"
            else:
                str_hash=str_hash+"0"
    return(str_hash)

def Hash_compare(Hash1,Hash2):
    n=len(Hash1)
    dev=0
    for i in range(n):
        if Hash1[i] !=Hash2[i]:
            dev=dev+1
    return(dev)
src_img1="lenna.png"
src_img2="lenna_noise.png"
Hash1=average_Hash(src_img1,8,8)
Hash2=average_Hash(src_img2,8,8)
Hash3=deviation_Hash(src_img1,9,8)
Hash4=deviation_Hash(src_img2,9,8)
print(Hash1,Hash2)
print(Hash3,Hash4)
print(Hash_compare(Hash1,Hash2))
print(Hash_compare(Hash3,Hash4))