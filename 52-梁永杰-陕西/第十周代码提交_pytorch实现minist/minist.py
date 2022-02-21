import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms  # 图像预处理包

class MnistNet(torch.nn.Module):   # 构建训练网络
    def __init__(self):
        super(MnistNet,self).__init__()
        self.fc1 = torch.nn.Linear(28*28,512)
        self.fc2 = torch.nn.Linear(512,100)
        self.fc3 = torch.nn.Linear(100,10)

    def forward(self,x):  # 前向推理 连接过程
        x = x.view(-1,28*28) # 就是reshape,第一个维度保持其原本维度不变
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x),dim=1)
        return x


def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),            # 将【H，W，C】张量变为【C,H,W】张量
         transforms.Normalize([0,],[1,])]  # 用均值和标准差归一化张量图像
    ) # 图像预处理 Compose把多个步骤整合到一起
    trainset = torchvision.datasets.MNIST(root='./data',train=True,
                                          download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,         #
                                              shuffle=True,num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data',train=False,
                                         download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,
                                             num_workers=2)                   #
    return trainloader,testloader

class Model:
    def __init__(self,net,cost,optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

    def create_cost(self,cost):
        support_cost = {
            'CROSS_ENTROPY' : torch.nn.CrossEntropyLoss(),
            'MSE' : torch.nn.MSELoss()
        } # 采用字典
        return support_cost[cost] # 输入键选择对应的损失函数

    def create_optimizer(self,optimist,**rests):
        support_optim = {
            'SGD'  : torch.optim.SGD(self.net.parameters(),lr=0.1,**rests),
            'ADAM' : torch.optim.Adam(self.net.parameters(),lr=0.01,**rests),
            'RMSP' : torch.optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train(self,train_loader,epoches = 3):   # 默认循环3代
        for epoch in range(epoches):
            running_loss = 0.0
            for i,data in enumerate(train_loader,0):
                inputs, labels = data
                self.optimizer.zero_grad()                    # 梯度初始化 0
                # 训练过程
                outputs = self.net(inputs)
                loss = self.cost(outputs,labels)              # 损失函数计算
                loss.backward()                               # 反响传播
                self.optimizer.step()                         # 优化

                running_loss += loss.item()                   # 记录loss
                if i % 100 == 0:  # 每100次迭代进行输出
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1,(i + 1)*1./len(train_loader)*100,running_loss /100))
                    running_loss = 0.0   # 重新初始化loss ，进行下一波计算

        print('Finished Training')

    def evaluate(self,test_loader):
        print('Evaluating...')
        correct = 0
        total = 0
        with torch.no_grad(): # 测试和预测时没有梯度  它将减少原本需要_grad=True的计算的内存消耗。
        # 在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。
        # 而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():，
        # 强制之后的内容不进行计算图构建。

            for data in test_loader:
                images,labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Acc of the network on the test images: %d %%' % (100 * correct / total))



if __name__  == '__main__':
    net = MnistNet()
    model = Model(net,'CROSS_ENTROPY','RMSP')
    train_loader,test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)