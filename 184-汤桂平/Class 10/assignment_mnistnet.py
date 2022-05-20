# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/5/2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self, net, cost, optimist):
        self.net = net   #在模型中定义网络
        self.cost = self.create_cost(cost)  #模型中定义损失函数
        self.optimizer = self.create_optimizer(optimist)  #定义优化器

    def create_cost(self,cost):
        cost_set = {'CROSS_ENTROPY':torch.nn.CrossEntropyLoss(), 'MSE': torch.nn.MSELoss()}
        return cost_set[cost]

    def create_optimizer(self, optimist, **rests):  #带双星号（**）的参数不传值时默认是一个空的字典
        optim_set = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return optim_set[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0  #对每一个epoch初始化损失函数
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()  # 清除优化器中的梯度信息，否则会被累积

                # forward + backward + optimize
                outputs = self.net(inputs)  # 正向计算
                loss = self.cost(outputs, labels)  # 计算损失函数
                loss.backward()  # 反向过程，计算loss对各参数的梯度
                self.optimizer.step()  # 更新参数

                '''
                一个epochs里按照很多个batchs进行训练。所以需要把一个epochs里的每次的batchs的loss加起来，
                等这一个epochs训练完后，会把累加的loss除以batchs的数量，得到这个epochs的损失
                '''
                running_loss += loss.item()  # 获取loss=tensor元素数值（具体、精确数值，而不是tensor）,计算损失。
                if i % 100 == 0:  # 每100个batch打印状态
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                            (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict 测试和推理时禁用梯度更新，以节省内存
            for data in test_loader:
                images, labels = data  # data=trainloader由inputs=图像和labels组成

                outputs = self.net(images)  # 得到测试结果，共10个输出
                predicted = torch.argmax(outputs, 1)  # 张量output在dim=1维的最大值的索引，即找到概率最大的那个
                total += labels.size(0)
                '''
                预测值与标签值相同的个数，当predicted和labels为张量时，predicted == labels返回的是一个张量，
                对应位置元素相等时为1，反之为0，求sum()则把这些元素求和，故(predicted == labels).sum()返回的也是
                一个张量，其元素只有一个，为预测值与标签值相同的个数。而item()函数的作用取出张量中的元素值。
                '''
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    '''
    torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
    transforms.ToTensor()把灰度范围从PIL图像的0-255变换到Torch.tensor的0-1之间
    transforms.Normalize(mean = [a, b, c], std = [d, e, f])表示利用公式(image-mean)/std
    把数据归一化处理。
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize([0, ], [1, ])])
    '''
    torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
    root：存在MNIST/processed/training.pt和MNIST/processed/test.pt的数据集的根目录
    train：（bool，可选）–如果为True，则从training.pt创建数据集，否则从test.pt创建数据集
    transform：（可调用，可选）–接受PIL图像并返回已转换版本的函数/转换
    target_transform：（可调用，可选）–接受目标并对其进行转换的函数/转换
    download：（bool，可选）–如果为true，则从internet下载数据集并将其放在根目录中。如果数据集已下载，则不会再次下载
    '''
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    '''
    torch.utils.data.DataLoader(dataset, batch_size=1, shuffle= False, sampler=None, batch_sampler=None, num_workers=0
    collate_fn=<function default_collate), pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
    数据加载器: 结合了数据集和取样器，并且可以提供多个线程处理数据集。在训练模型时使用到此函数，用来把训练数据分成多个小组，
    此函数每次抛出一组数据，直至把所有的数据都抛出。就是做一个数据的初始化。
    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
                the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id as input, after seeding and before data
            loading. (default: None)

    '''
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader



class Mnistnet(torch.nn.Module):
    def __init__(self):
        '''

        super(SubClass, self).method() 的意思是，根据self去找SubClass的‘父亲’，然后调用这个‘父亲’的method()
        torch.nn.Linear(in_features, out_features, bias=True) 函数是一个线性变换函数:y = x * A + b
        其中，in_features为输入样本的大小，out_features为输出样本的大小，bias默认为true。
        如果设置bias = false那么该层将不会学习一个加性偏差。Linear()函数通常用于设置网络中的全连接层.
        '''
        super(Mnistnet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        '''
        在PyTorch中view()函数作用为重构张量的维度，相当于numpy中的resize()的功能，参数为-1时需要自行判断。
        '''
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)   #dim=1为按行求softmax
        return x

if __name__ == '__main__':
    net = Mnistnet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()  #加载训练数据和测试数据
    model.train(train_loader)   #网络训练
    model.evaluate(test_loader)  #推理

