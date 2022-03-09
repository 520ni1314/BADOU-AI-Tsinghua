# from tensorflow.keras.datasets import  mnist
# import matplotlib.pyplot as plt
#
# import numpy as np
# # 1.下载训练数据和测试数据
# # 2.显示测试数据集中的一张图片
# (train_images,train_labels),(test_images,test_labels) = mnist.load_data()
#
# # print("train_images",train_images.shape)
# # print("train_labels",train_labels)
# # print("test_image_shape",test_images.shape)
# # print("test_labels",test_labels)
# #
# # digit = train_images[0]
# # plt.imshow(digit,cmap=plt.cm.binary)
# # plt.imshow(digit,cmap=plt.cm.binary)
# # plt.show()
#
# from tensorflow.keras import  models
# from  tensorflow.keras import  layers
#
# # 构建网络模型
# network = models.Sequential()
# #layers 添加网络层  ，dense ： 全连接层
# # layers.Dense（）： 构造一个数据处理层
# # input_shape :输入28*28 二维数组，后面任意
# network.add(layers.Dense(512,activation="relu",input_shape=(28*28,)))
# network.add(layers.Dense(10,activation="softmax"))
# network.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuray"])
# # 优化函数  optimizer
# # loss 函数  交叉熵
# # metrics  准确度
# network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
# #
# # # 数据处理，将训练和测试数据变形并归一化
# # 将原来的60000 个28*28 的数组 ，转化成 一个60000*28*28 的数组
# train_images = train_images.reshape((60000,28*28))
# train_images = train_images.astype("float32") /255
# print(train_images.shape)
#
# test_images = test_images.reshape((10000,28*28))
# test_images = test_images.astype("float32")/255
# print(test_images.shape)
#
# from tensorflow.keras.utils import  to_categorical
#
# #
# # 将图片的标签做处理 one_hot
# #
# print("before",test_labels[1])
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# print("after",test_labels[1])
#
# # 将训练数据投入到网络模型中
# network.fit(train_images,train_labels,epochs = 5,batch_size=128)
#
# #查看测试数据的准确度
# test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=1)
# print(test_loss)
# print("test_acc",test_acc)
#
# # 加载手写数字识别的数据（训练和测试）
# (train_images,train_labels),(test_images,test_labels) = mnist.load_data()
# # 显示测试数据中一张图片,并验证其标签值
# digit = test_images[2]
# plt.imshow(digit,cmap=plt.cm.binary)
# plt.show()
#
# test_images = test_images.reshape((10000,28*28))
# res = network.predict(test_images)
# print(res.shape)
# #
# for i in range(res[2].shape[0]):
#     if (res[2][i]==1):
#         print("the number for the picture is:",i)
#         break
#
# #

from tensorflow.keras.datasets import  mnist
import matplotlib.pyplot as plt

#1. 加载训练数据和测试数据
(train_datas,train_labels),(test_datas,test_labels) = mnist.load_data()
# print("train_datas",train_datas.shape) # (60000,28,28)
# print("train_labels",train_labels)     # [5 0 4 ... 5 6 8]
# print("test_datas",test_datas.shape)   # (10000,28,28)
# print("test_labels",test_labels)       #  [7 2 1 ... 4 5 6]

# 2.构建网络模型
# 导入模型和层
from  tensorflow.keras import models
from  tensorflow.keras import  layers
net = models.Sequential()
net.add(layers.Dense(512,activation='relu',input_shape=(28*28,))) # h,w,c
net.add(layers.Dense(10,activation='softmax'))
net.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
#3. 数据处理(形状和归一化)
train_datas = train_datas.reshape((60000,28*28))
train_datas = train_datas.astype("float32")/255

test_datas = test_datas.reshape((10000,28*28))
test_datas = test_datas.astype("float32") /255

# 4。将标签做one-hot 处理
from tensorflow.keras.utils import to_categorical

# labels = [3,4,5]
# labels2 = [2,4]
# cate1 = to_categorical(labels)
# cate2 = to_categorical(labels2)
# print(cate1)
# print(cate2)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 5.开始训练数据
net.fit(train_datas,train_labels,epochs=5,batch_size=128)

# 6 .输入测试数据，查看效果
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
loss,acc = net.evaluate(test_datas,test_labels,verbose =2)
print(loss)
print(acc)


#7 将一张手写图像输入中
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = net.predict(test_images)
print(res.shape)

for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print("the number for the picture is : ", i)
        break
