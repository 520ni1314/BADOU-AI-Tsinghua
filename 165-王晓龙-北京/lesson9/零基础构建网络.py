import  numpy
import  scipy.special
class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        # 初始化网络、设置输入层，中间层、输出层节点数,这样就能决定网络的形状和大小
        # 根据输入参数来设置动态网络的形态，所以不把参数写死
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #设置学习率
        self.lr = learningrate
        pass
        # 初始化权重，把权重初始化为-0.5 到0.5
        # w1 是输入层和中间层节点间链路权重形成的矩阵
        # w2 是中间层和输出层间链路权重形成的矩阵
        self.w1 = numpy.random.rand(self.hnodes,self.inodes) - 0.5
        self.w2 = numpy.random.rand(self.onodes,self.hnodes) - 0.5

        # self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        # self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # sigmod 函数在python 中可以直接调用，
        # scipy.special.expit 对应的是sigmod 函数
        self.activation_function = lambda x:scipy.special.expit(x)
    def train(self,input_list,targets_list):
        # 根据输入的训练数据更新节点权重,训练数据
        # input_list： 输入训练数据 ，targets_list ： 训练数据对应的正确结果
        # 将input_list 和targets_list 转置
        inputs = numpy.array(input_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.w1,inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.w2,hidden_outputs)
        # 经过激活函数输出最终结果
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.w2.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.w2 += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.w1 += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass
    def query(self,inputs):
        # 主要实现，接收输入数据，经过神经网络的层层计算，输出最终结果
        # 可以理解成 推理过程
        hidden_inputs = numpy.dot(self.w1,inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算最外层接收的信号量
        final_inputs = numpy.dot(self.w2,hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_output = self.activation_function(final_inputs)
        # print("input",inputs)
        # print("hidden_inputs",hidden_inputs)
        # print("hidden_outputs",hidden_outputs)
        # print("final_inputs",final_inputs)

        print("final_output",final_output)
        return final_output

#2 初始化网络和学习率
# 由于一张图片总有28*28 =784 个数值，因此需要网络的输入具备784个输入的节点

input_nodes = 784
hidden_nodes = 200
output_nodes =10

learning_rate = 0.1  # 学习率
net = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)
# net.query([1.0, 0.5, -1.5])


#3  读取训练数据和测试数据
trainning_data_file = open("dataset/mnist_train.csv","r") # 以读的方式
trainning_data_list = trainning_data_file.readlines()
trainning_data_file.close()


# 4.
# 加入epochs ,设定网络的训练循环次数

epochs = 5
for epoch in range(epochs):
    # 把数据用“，” 区分，
    for record in trainning_data_list:
        all_values = record.split(",")
        inputs = (numpy.asfarray(all_values[1:]))/255.0*0.99 +0.01
        # 设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] =0.99
        net.train(inputs,targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

#5 将测试图片输入网络，看看检测的效果如何
score = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为：",correct_number)

    # 预处理数字图片
    inputs = (numpy.asfarray(all_values[1:]))/255.0 *0.99 +0.01
    # 让网络判断图片对应的数字，推理
    outputs = net.query(inputs)
    # 找到数值最大的神经元对应的编号
    label= numpy.argmax(outputs)
    print("output reslut is :",label)
    if label == correct_number:
        score.append(1)
    else:
        score.append(0)
print(score)

# 计算图片的成功率
scores_array = numpy.asarray(score)
print("perfermance = ",scores_array.sum()/scores_array.size)




