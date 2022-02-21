import numpy
import numpy as np
import scipy.special

class NeuralNtework:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5), (self.hnodes,self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
    def train(self,inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        ###由损失更新权重系数
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
        pass


    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

inputnodes = 784
hiddennodes = 200
outputnodes = 10
learningrate = 0.1
n = NeuralNtework(inputnodes, hiddennodes, outputnodes, learningrate)
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        targets = numpy.zeros(outputnodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
test_data_file = open("mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    test_values = record.split(',')
    correct_numbers = int(test_values[0])
    print("该图片对应的数字是：", correct_numbers)
    test_inputs = (np.asfarray(test_values[1:]))/255.0 * 0.99 + 0.01
    test_outputs = n.query(test_inputs)
    label = np.argmax(test_outputs)
    print("网络预测的数字是：", label)
    if label == correct_numbers:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
scores_array = np.asarray(scores)
print("scores perfermance:", scores_array.sum() / scores_array.size)
# writer=tf.summary.FileWriter('logs', tf.get_default_graph())
# writer.close()
