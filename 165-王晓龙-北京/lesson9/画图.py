import numpy
import matplotlib.pyplot as plt
#%matplotlib inline

# open 按路径打开一个文件
# data_file = open("dataset/mnist_test.csv")
# data_list = data_file.readlines() # 按行读入 10 行数据
# data_file.close()
# # print(data_list)
# # print(len(data_list))
# print(len(data_list[0])) # 第一行
# print(data_list[0])

data_file = open("dataset/test.csv")
# readlines()  按行读取所有的行，包括换行符号一块读取
# 返回： 一个字符串函数['a,b,c,c,c,cer,sdf\n', '1,2,3,5,3,6,8\n', '3,4,1,6,4,2,5\n']
data_list = data_file.readline()
data_file.close()
print(data_list)
print(data_list[0])
# all_values = data_list[0].split(",")
# print(all_values)
# image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
# print(image_array)
# all_values = all_values[1:].reshape((28,28))
# image_array  = numpy.asfarray()


