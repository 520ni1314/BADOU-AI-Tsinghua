import numpy
import scipy.special

class NerualNetWork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learningrate):
        self.inodes=input_nodes
        self.hnodes=hidden_nodes
        self.onodes=output_nodes
        self.lr=learningrate

        self.wih=numpy.random.rand(self.hnodes,self.inodes)-0.5
        self.who=numpy.random.rand(self.onodes,self.hnodes)-0.5

        #定义激活函数
        self.activation_funtion=lambda x:scipy.special.expit(x)

        pass

    def train(self,input_list,target_list):
        inputs=numpy.array(input_list,ndmin=2).T
        target=numpy.array(target_list,ndmin=2).T

        #数据从输入层经过隐藏层到达输出层
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_funtion(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_funtion(final_inputs)

        #计算误差
        output_error=target-final_outputs
        hidden_error=numpy.dot(self.who.T,output_error*final_outputs*(1-final_outputs))
        self.who+=self.lr*numpy.dot(output_error*final_outputs*(1-final_outputs),numpy.transpose(hidden_outputs))
        self.wih+=self.lr*numpy.dot(hidden_error*hidden_outputs*(1-hidden_outputs),numpy.transpose(inputs))

        pass

    def query(self,inputs):
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_funtion(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_funtion(final_inputs)
        # print(final_outputs)
        return final_outputs

#设定初始化参数
input_nodes=28*28
hidden_nodes=300
output_nodes=10
learningrate=0.5
net=NerualNetWork(input_nodes,hidden_nodes,output_nodes,learningrate)

#读入训练数据
import numpy

training_data_file=open("dataset/mnist_train.csv")
training_data_list=training_data_file.readlines()
training_data_file.close()
# print(training_data_list)
# print(type(training_data_list))

epochs=10
for e in range(epochs):
    for record in training_data_list:
        all_values=record.split(",")
        # print(all_values)
        inputs=(numpy.asfarray(all_values[1:]))/255*0.99+0.01
        #onehot
        targets=numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        net.train(inputs,targets)

#导入验证集数据
test_data_file=open("dataset/mnist_test.csv")
test_data_list=test_data_file.readlines()
test_data_file.close()
score=[]
#验证
for record in test_data_list:
    all_values=record.split(",")
    correct_num=int(all_values[0])
    inputs=numpy.asfarray(all_values[1:])/255*0.99+0.01
    outputs=net.query(inputs)
    print(outputs)
    label=numpy.argmax(outputs)    #numpy.argmax()，输出最大值的位置(index)
    # print(label)
    print("正确结果：",correct_num,"预测结果：",label)

    if label==correct_num:
        score.append(1)
    else:
        score.append(0)
print(score)

#计算神经网络的成功率
scores_array=numpy.asarray(score)
print("神经网络的正确率为：",scores_array.sum()/len(scores_array))

