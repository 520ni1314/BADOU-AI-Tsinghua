from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from numpy import *
import numpy as np

X = np.array([[1, 1],
              [2, 2]]

             )
K = np.dot(X,X)

print(K)