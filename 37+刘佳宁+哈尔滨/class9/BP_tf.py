#############################
"""
  用Tensorflow框架搭建BP神经网络
  配置： tf123
"""
#############################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 建立数据集, y = x^2, x, noise, y_data -> 200*1,
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
# 引入高斯噪声noise
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise
# print(x_data.shape,noise.shape,y_data.shape)

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

# 定义BP神经网络的中间层, Weights_L1, biases_L1 -> tanh(Weights_L1 * x + biases_L1)
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义BP神经网络的输出层, Weights_L2, biases_L2 -> tanh(Weights_L2 * L1 + biases_L2)
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))

# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 建立一个session训练BP神经网络
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())

    # 开始训练，迭代2000次
    for i in range(2000):
        sess.run(train_step,feed_dict = {x:x_data,y:y_data})

    # 得到预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})

    # 得出结果并做图
    print(y_data.shape,prediction_value.shape)
    plt.figure()
    # 绘制散点图（真实值）
    plt.scatter(x_data,y_data)
    # 绘制曲线图（预测值）
    plt.plot(x_data,prediction_value,'r-',lw = 5)
    plt.show()