# -- coding:utf-8 --
import numpy
import matplotlib.pyplot as plt
import numpy as np

import week9_homework.NeuralNetwork_init as NN

data_file = open("./dataset/mnist_test.csv")
test_data_list = data_file.readlines()
data_file.close()




# 初始化网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
epochs = 100

net = NN.NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# 获取训练数据
train_data_file = open("./dataset/mnist_train.csv")
train_data_list = train_data_file.readlines()
train_data_file.close()

for e in range(epochs):
    for record in train_data_list:
        all_values = record.split(",")
        inputs = np.asfarray(all_values[1:])/255.0*0.99 + 0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        net.train(inputs,targets)

# 测试
scores = []
for record in test_data_list:
    all_values = record.split(",")
    correct_number = int(all_values[0])
    print("该图片对应的数字为：",correct_number)
    #预处理数字图片
    inputs = np.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
    output_puts = net.query(inputs)
    label = np.argmax(output_puts)
    print("output result is :",label)

    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
# 计算图片判断的准确率

acc = sum(scores) / len(scores)
print("perfermance = ",acc)
