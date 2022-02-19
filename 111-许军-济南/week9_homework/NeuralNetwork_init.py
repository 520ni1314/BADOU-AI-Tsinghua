# -- coding:utf-8 --
import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.innodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 设置学习率
        self.lr = learningrate
        # 初始化权重
        self.wih = np.random.rand(self.hnodes,self.innodes) - 0.5
        self.who = np.random.rand(self.onodes,self.hnodes) - 0.5

        # sigmoid函数
        self.activation_function = lambda x:scipy.special.expit(x)
    def train(self,inputs_list,targets_list):
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets -final_outputs
        hidden_errors = np.dot(self.who.T,output_errors*final_outputs*(1-final_outputs))# 激活函数求导f（x）*(1-f(x))
        self.who -= self.lr * np.dot(output_errors*final_outputs*(1-final_outputs),np.transpose(hidden_outputs))
        self.wih -= self.lr * np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),np.transpose(inputs))


    def query(self,inputs):
        # 中间层的结果
        hidden_inputs = np.dot(self.wih,inputs)
        # 激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出结果
        final_inputs = np.dot(self.who,hidden_outputs)
        # 激活后结果
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


