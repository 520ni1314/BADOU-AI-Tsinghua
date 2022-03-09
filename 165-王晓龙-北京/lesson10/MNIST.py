# 导入包
import  torch
import torch.nn as nn
import  torch.optim as optim
import  torch.nn.functional as F
import  torchvision
import  torchvision.transforms  as transforms

# 自定义模型类
class Model:

    # 初始化方法
    # 输入网络模型，损失函数，优化项
    def __init__(self,net,cost,optimist):
        self.net = net
        self.cost = self.create_cost(cost) #返回方法中的值赋值给cost 属性
        self.optimizer = self.create_optimizer(optimist)
    # 损失函数方法
    def create_cost(self,cost):
        # 损失函数可选交叉熵和均值平方，还可以自己添加，按格式key -value
        support_cost ={
            "CROSS_ENTROPY":nn.CrossEntropyLoss(), #交叉熵
            "MSE":nn.MSELoss                       # 均值平方
        }
        return support_cost[cost]
    # 优化项选择方法
    # 最后的参数是预留
    def create_optimizer(self,optimist,**rests):
        support_optim ={
            # 优化网络中的参数，学习率，待定参数
            "SGD": optim.SGD(self.net.parameters(),lr=0.1,**rests),
            "ADAM": optim.Adam(self.net.parameters(), lr=0.01, **rests),
            "RMSP": optim.RMSprop(self.net.parameters(), lr=0.001, **rests)

        }
        return support_optim[optimist]
    # 训练方法
    def train(self,train_loader,epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i,data in enumerate(train_loader,0):
                inputs,labels = data
                self.optimizer.zero_grad() # 梯度清零

                outputs = self.net(inputs)
                loss = self.cost(outputs,labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

                if i %100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' % (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print("Finished Training")

    # 推理 方法
    def evaluate(self,test_loader):
        print('Evaluating ...')
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

# 定义数据加载函数
def mnist_load_data():
      # 将数据转化成tensor ,服从正态分布
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0,],[1,])])
    # 加载训练集和测试集，

    trainset = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size =32,shuffle = True,num_workers=2)
    testset = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,num_workers=2)
    return  trainloader,testloader

# 定义数字识别类
class MnistNet(torch.nn.Module):
    # 初始化,需要训练的写到初始化
    def __init__(self):
        super(MnistNet,self).__init__()
        self.fc1 = torch.nn.Linear(28*28,512) # 第一层输入28*28，512
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512,10)
    # 前向计算 。不需要训练
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x


# 实例模型，并调用其中的方法
if __name__ == '__main__':
    net = MnistNet()  # 实例化网络模型
    model = Model(net,'CROSS_ENTROPY','RMSP')
    train_loader,test_loader = mnist_load_data() # 加载数据
    model.train(train_loader) #训练
    model.evaluate(test_loader) # 推理



