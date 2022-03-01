#############################
"""
  用Keras框架搭建BP神经网络
  完成mnist数据集的手写识别
  配置： tf123
"""
#############################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# 从框架内提取出mnist数据集信息，包括训练图片标签和测试图片标签
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels = ', test_labels)

# 建立神经网络network有效识别手写数字图片, 中间层28*28 -> 512, 输出层512 -> 10
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

# 对图片进行预处理并归一化处理, 60000*28*28 -> 60000*784, 10000*28*28 -> 10000*784
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255
print(train_images.shape,test_images.shape)

# 对标签进行归一化预处理, 1*1 -> 1*10, 如7 -> [0,0,0,0,0,0,0,1,0,0]
train_labels = to_categorical(train_labels)
test_labels_new = to_categorical(test_labels)
print("before change:" ,test_labels[0],'\n',"after change: ", test_labels_new[0])
test_labels = test_labels_new

# 将数据输入网络中进行训练, epochs = 5, batchsize = 128
network.fit(train_images, train_labels, epochs=5, batch_size = 128)

# 网络训练好后，输入测试集得到损失值和正确率
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss: ', test_loss)
print('test_acc: ', test_acc)

# 输入随机一张手写数字图片，查看识别结果
# 显示图片
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
num = np.random.randint(0,10000)
# print(num)
digit = test_images[num]
plt.figure()
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
# 将图片预处理，输入网络模型，得到输出结果
digit = digit.reshape((1,28*28))
res = network.predict(digit)
print(res)
# 将长向量转化为识别数字输出
for i in range(res.shape[1]):
    if (res[0][i] == 1):
        print("the number for the picture is : ", i)
        break