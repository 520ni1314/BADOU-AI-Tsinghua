import numpy as np
import math
import cv2
import copy


def bgr2gray(img):

    ori_img = img
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gray_img = r*0.3 + g*0.59 + b*0.11
    gray_img = gray_img.astype(np.uint8)
    
    return gray_img

def gaussian_fun(sigma,img):

    dim = round(5*sigma) + 1
    dim = dim+1 if dim%2==0 else dim
    tmp = np.arange(dim) - dim//2
    tmpx = np.tile(tmp,(dim,1))
    tmpy = tmpx.T

    n1 = -1/(2*sigma**2)
    n2 = 1/(2*math.pi*sigma**2)
    gaussian_filter = n2*np.exp(n1*(tmpx**2+tmpy**2))
    gaussian_filter = gaussian_filter/gaussian_filter.sum()
    
    gau_img = np.zeros((img.shape))
    pad_tmp = dim//2
    pad_img = np.pad(img,((pad_tmp,pad_tmp),(pad_tmp,pad_tmp)),'constant')
    dy, dx = img.shape

    for y in range(dy):
        for x in range(dx):
            gau_img[y,x] = np.sum(pad_img[y:y+dim,x:x+dim]*gaussian_filter)

    return gau_img
    
def sobel_tidu(gau_img):

    gau_img = gau_img
    dy, dx = gau_img.shape
    img_tidux = np.zeros((dy,dx))
    img_tiduy = copy.copy(img_tidux)
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    pad_img = np.pad(gau_img,((1,1),(1,1)),'constant')
    for y in range(dy):
        for x in range(dx):
            img_tidux[y,x] = np.sum(pad_img[y:y+3,x:x+3]*sobel_x)
            img_tiduy[y,x] = np.sum(pad_img[y:y+3,x:x+3]*sobel_y)
    img_tidu = np.sqrt(img_tidux**2+img_tiduy**2)

    return img_tidu, img_tidux, img_tiduy

def nms_fun(img_tidu, img_tidux, img_tiduy):

    img_tidu = img_tidu
    img_tidux[img_tidux==0] = 0.000001
    img_k = img_tiduy/img_tidux
    dy, dx = img_k.shape
    img_bian = np.zeros((dy,dx))
    for y in range(1,dy-1):
        for x in range(1,dx-1):
            tmp = img_tidu[y-1:y+2,x-1:x+2]
            flag = True

            if img_k[y,x]<-1:
                num_1 = (tmp[0,1]-tmp[0,0])/img_k[y,x] + tmp[0,1]
                num_2 = (tmp[2,1]-tmp[2,2])/img_k[y,x] + tmp[2,1]
                if not (img_tidu[y,x]>num_1 and img_tidu[y,x]>num_2):
                    flag = False
            elif img_k[y,x]>1:
                num_1 = (tmp[0,2]-tmp[0,1])/img_k[y,x] + tmp[0,1]
                num_2 = (tmp[2,1]-tmp[2,0])/img_k[y,x] + tmp[2,1]
                if not (img_tidu[y,x]>num_1 and img_tidu[y,x]>num_2):
                    flag = False
            elif img_k[y,x]<=1 and img_k[y,x]>=0:
                num_1 = (tmp[0,2]-tmp[1,2])*img_k[y,x] + tmp[1,2]
                num_2 = (tmp[1,0]-tmp[2,0])*img_k[y,x] + tmp[1,0]
                if not (img_tidu[y,x]>num_1 and img_tidu[y,x]>num_2):
                    flag = False
            elif img_k[y,x]>=-1 and img_k[y,x]<0:
                num_1 = (tmp[1,0]-tmp[0,0])*img_k[y,x] + tmp[1,0]
                num_2 = (tmp[2,2]-tmp[1,2])*img_k[y,x] + tmp[1,2]
                if not (img_tidu[y,x]>num_1 and img_tidu[y,x]>num_2):
                    flag = False
            if flag:
                img_bian[y,x] = img_tidu[y,x]

    return img_bian

def threshold_edge(img_bian,low_th,high_th):
    
    img_bian = img_bian
    low_th = low_th
    high_th = high_th
    dy, dx = img_bian.shape
    zhan = []
    for y in range(1,dy-1):
        for x in range(1,dx-1):
            if img_bian[y,x]>=high_th:
                img_bian[y,x] = 255
                zhan.append((y,x))
            elif img_bian[y,x]<=low_th:
                img_bian[y,x] = 0

    while not len(zhan)==0:
        y, x = zhan.pop()
        tmp = img_bian[y-1:y+2,x-1:x+2]
        for yj in [0,1,2]:
            for xi in [0,1,2]:
                if tmp[yj,xi]>low_th and tmp[yj,xi]<high_th:
                    m, n = y-(1-yj), x-(1-xi)
                    img_bian[m,n] = 255
                    zhan.append((m,n))
               
    for y in range(dy):
        for x in range(dx):
            if img_bian[y,x]!=0 and img_bian[y,x]!=255:
                img_bian[y,x] = 0
    
    return img_bian






if __name__=='__main__':
    
    img = cv2.imread('../lenna.png')
    gray_img = bgr2gray(img)
    sigma = 0.5
    gau_img = gaussian_fun(sigma,gray_img)
    img_tidu, img_tidux, img_tiduy = sobel_tidu(gau_img)
    img_bian = nms_fun(img_tidu,img_tidux,img_tiduy)
    low_th = img_bian.mean()*0.5
    high_th = low_th*3
    canny_img = threshold_edge(img_bian,low_th,high_th)

    cv2.imshow('gray',gray_img)
    cv2.imshow('gau',gau_img.astype(np.uint8))
    cv2.imshow('bian',img_bian.astype(np.uint8))
    cv2.imshow('canny',canny_img)
    cv2.waitKey(0)
