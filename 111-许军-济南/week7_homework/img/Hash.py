# -- coding:utf-8 --
import cv2
def aHash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash = ""
    sum = 0
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            sum += gray[i][j]
    avg = sum / 64
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i][j] > avg:
                hash += "1"
            else:
                hash += "0"
    return hash

def dHash(img):
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash = ""
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash += "1"
            else:
                hash += "0"
    return hash

def cmpHash(hash1,hash2):
    if len(hash1) != len(hash2):
        return -1
    n = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n
if __name__ == '__main__':
    img = cv2.imread("./img/lenna.png")
    img2 = cv2.imread("./img/lenna_noise.png")
    hash1 = aHash(img)
    hash2 = dHash(img2)
    simi = cmpHash(hash1,hash2)
    print("均值哈希的相似度为：{}".format(simi))
