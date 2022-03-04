#################################
'''
    mnist数据集手写识别-pytorch
    环境：shunli(pytorch)
'''
#################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 下载mnist数据集
def mnist_load_data():
    # 创建一个容器进行数据类变换
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0,],[1,])])

    trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = True, num_workers = 2)
    testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle = True, num_workers = 2)

    return trainloader, testloader

# 创建手写数字识别模型
class Model:
    # 初始化,需要参数：net, cost, optimist
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

        pass

    # 损失函数的选择
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    # 优化器的选择
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    # 训练模型
    def train(self, train_loader, epochs = 2):
        for epoch in range(epochs):
            # 设置初始损失值running_loss为0
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # 前向传播+后向传播+优化
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training!')

    # 测试模型
    def evaluate(self, test_loader):
        print('Evaluating......')
        # 初始化
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# 建立神经网络
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim = 1)

        return x

# 利用训练好的网络模型输出识别结果
if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)