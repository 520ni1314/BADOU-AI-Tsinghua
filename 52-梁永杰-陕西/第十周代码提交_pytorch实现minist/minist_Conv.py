import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 将【H，W，C】张量变为【C,H,W】张量
         transforms.Normalize([0, ], [1, ])]  # 用均值和标准差归一化张量图像
    )
    trainset = torchvision.datasets.MNIST(root='./data',train=True,
                                          download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,         #
                                              shuffle=True,num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data',train=False,
                                         download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,
                                             num_workers=2)                   #
    return trainloader,testloader


class Minist_net(torch.nn.Module):
    def __init__(self):
        super(Minist_net,self).__init__()
        self.layer1 = torch.nn.Sequential(
           torch.nn.Conv2d(1,10,kernel_size=3,stride=1,padding=2),
           torch.nn.BatchNorm2d(10),   # 对数据进行归一化
           torch.nn.ReLU(inplace=True),
           torch.nn.MaxPool2d(kernel_size=3,stride=2),
           torch.nn.Conv2d(10,5,kernel_size=5, stride=1, padding=2),
           torch.nn.BatchNorm2d(5),
           torch.nn.ReLU(inplace=True),
           torch.nn.MaxPool2d(kernel_size=2,stride=2),
           )
        self.layer2 = torch.nn.Sequential(
           torch.nn.Linear(245,1024),
           torch.nn.ReLU(inplace=True),
           torch.nn.Linear(1024,128),
           torch.nn.ReLU(inplace=True),
           torch.nn.Linear(128,10)
           )

    def forward(self,x):
        x = self.layer1(x)
        x = x.view(x.size(0),-1)
        x = self.layer2(x)
        return x

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
        train_loss = []                         # 记录每次迭代的loss
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

                train_loss.append(loss.item())  # 保存每个batch的损失
                running_loss += loss.item()                   # 记录loss
                if i % 100 == 0:  # 每100次迭代进行输出
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1,(i + 1)*1./len(train_loader)*100,running_loss /100))
                    running_loss = 0.0   # 重新初始化loss ，进行下一波计算
        plt.figure()
        plt.plot(train_loss)
        plt.title("Train loss ")
        plt.show()
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
    net = Minist_net()
    model = Model(net,'CROSS_ENTROPY','RMSP')
    train_loader,test_loader = mnist_load_data()

    model.train(train_loader)
    model.evaluate(test_loader)
