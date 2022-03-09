import random
import numpy as np
import cv2
import time
def saltpeppernoise(src,persantage):

    img = src
    h, w = img.shape
    per_w = int(w*persantage)
    xlist = [i for i in range(w)]
    for y in range(h):
        xlist = random.sample(xlist,per_w)
        for x in xlist:
            if random.choice([True,False]):
                img[y,x] = 255
            else:
                img[y,x] = 0
    return img


if __name__=='__main__':
    
    img = cv2.imread('../lenna.png')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayimg',gray_img)
    saltpepperimg = saltpeppernoise(gray_img,0.5)
    cv2.imshow('saltpepperimg',saltpepperimg)
    cv2.waitKey(0)
