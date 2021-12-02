import cv2
import numpy as np
import matplotlib.pyplot as plt

img_loc = r"02_math,graph_basic/lenna.png"


class ImageBasic:
    def __init__(self) -> None:
        pass


class ImageProcess(ImageBasic):
    def __init__(self, img_loc):
        super().__init__()
        self.__img = cv2.imread(img_loc)
        self.__h = self.__img.shape[0]
        self.__w = self.__img.shape[1]
        self.__c = self.__img.shape[2]

    @property
    def img(self):
        return self.__img

    def __EmptyImgGen(self, height, width, channel=None, type_=None):
        channel_st = self.__c if not channel else channel
        type_st = self.__img.dtype if not type_ else type_
        img = np.zeros(shape=(height, width, channel_st), dtype=type_st)
        return img
