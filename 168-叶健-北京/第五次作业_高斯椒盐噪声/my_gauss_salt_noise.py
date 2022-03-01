import random
import cv2
def noise_gauss(source,means,sigma,percent):
    noise_img=source.copy()
    height, weight = source.shape
    for i in range(int(height*weight*percent)):
        x=random.randint(0,height-1)
        y=random.randint(0,weight-1)
        noise_img[x][y]=noise_img[x][y]+random.gauss(means,sigma)
        noise_img[x][y]=noise_img[x][y] if 0<=noise_img[x][y] else 0
        noise_img[x][y]=noise_img[x][y] if noise_img[x][y]<=255 else 255
    return noise_img
def noise_salt(source,percent):
    noise_img = source.copy()
    height, weight = source.shape
    for i in range(int(height * weight * percent)):
        x = random.randint(0, height - 1)
        y = random.randint(0, weight - 1)
        noise_img[x][y]=0 if random.random()<0.5 else 255
    return noise_img
if __name__ == '__main__':
    img=cv2.imread("lenna.png",0)
    cv2.imshow("source",img)
    gauss_img=noise_gauss(img,2,4,0.8)
    cv2.imshow("gauss",gauss_img)
    salt_img=noise_salt(img,0.2)
    cv2.imshow("salt",salt_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()