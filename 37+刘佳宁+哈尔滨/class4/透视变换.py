import cv2
import numpy as np

img = cv2.imread('photo.jpg')
print(img.shape)
result = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
############################
# 通过已知的4组点得到变换矩阵
############################
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 生成透视变换矩阵m
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)

# 透视变换，并截取(0,0) -> (337,488)
result = cv2.warpPerspective(result, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
