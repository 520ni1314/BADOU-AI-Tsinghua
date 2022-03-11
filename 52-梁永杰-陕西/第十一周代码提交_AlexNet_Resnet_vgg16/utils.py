import numpy as np
import matplotlib.pyplot as plt
import cv2


'''数据处理接口'''

def load_image(path):
    '''加地图片，并将图片转换为以图片中心为中心，以最短边长为边的正方形图片'''
    img = plt.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])     # 选取前两项 h、w
    yy = int((img.shape[0]  - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    square_img = img[yy:yy + short_edge,xx:xx + short_edge] #  以最短边为基准边将图片变为一个边长为最短边的正方形图像，图像中心图中心
    return square_img

def resize_image(image,size):
    '''将多张图片输入改变其形状，返回一个ndarry的一个图片组'''
    images = []
    for img in image:
        img = cv2.resize(img,size)
        images.append(img)
    images = np.array(images)
    return images

def print_answer(argmax):
    with open('./data/model/index_word.txt','r',encoding='utf-8') as f:
       #  f.readlines()  只能读取一次，下次运行读取无效
        synset = [l.split(';')[1][:-1] for l in f.readlines()]

        return synset[argmax]

if __name__ == '__main__':
    print(print_answer(1))