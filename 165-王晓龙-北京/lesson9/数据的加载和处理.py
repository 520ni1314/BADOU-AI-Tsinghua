import  numpy as np
import  matplotlib.pyplot as plt

# # open 打开文件
# data_file = open("dataset/mnist_test.csv")
# data_list = data_file.readlines()
# data_file.close()
#
# # print(len(data_list)) # 10*785
# # print(data_list[0])
#
# all_value = data_list[0].split(",") #用逗号区分数据.第一张图片
# # print(all_value)
# # 将list 转化成数组，第一个值是图片表示的数字，要去掉
# image_array = np.asfarray(all_value[1:]).reshape((28,28)) #28*28
# # # print(image_array)
# # print(all_value[0])
#
# # target 第8个元素的值是0.99 ，代表图片上的数字是7
# nodes = 10
# target = np.zeros(nodes) +0.01
# print(target)
# target[int(all_value[0])] = 0.99
# print(target)

# 网络的初始化
input_nodes = 784
hidden_nodes = 100
output_nodes =10
learning_rate =0.3
net = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open("dataset/mnist_train.csv")
# 读入100张图片
training_list = training_data_file.readlines()  # 100*785
training_data_file.close()

# print(training_list)
# record 代表一张图片
for record in training_list:
    all_values = record.split(",") # 每张图片用逗号区分
    # 归一化
    # np.asfarray 转化成数组
    inputs = np.asfarray(all_values[1:])/255.0 *0.99 +0.01

    # 图片与label 的对应关系
    targets = np.zeros(10) +0.01
    targets[int(all_values[0])] = 0.99
    net.train(inputs,targets)  # 输入到训练中

#思路
"""
1.读取训练文件数据
2.区分训练数据和label,对数据归一化处理
3.将训练数据和label 输入到forward 中训练

4.读取测试数据
5.区分测试数据和label
6.
"""

