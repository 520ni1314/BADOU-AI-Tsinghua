# author: orea

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist��ֱ��ͼ���⻯
����ԭ�ͣ� equalizeHist(src, dst=None)
src��ͼ�����(��ͨ��ͼ��)
dst��Ĭ�ϼ���
'''

# ��ȡ�Ҷ�ͼ��
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# �Ҷ�ͼ��ֱ��ͼ���⻯
dst = cv2.equalizeHist(gray)
cv2.imwrite("lenna-equalizeHist-gray.png", dst)

# ֱ��ͼ���ӻ�
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

# ��ɫͼ����⻯,��Ҫ�ֽ�ͨ�� ��ÿһ��ͨ�����⻯
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# �ϲ�ÿһ��ͨ��
result = cv2.merge((bH, gH, rH))
cv2.imwrite("lenna-equalizeHist-rgb.png", result)
