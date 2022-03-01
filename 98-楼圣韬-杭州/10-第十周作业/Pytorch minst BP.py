import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os

class MNISTNET(torch.nn.Module):
    def __init__(self):
        super(MNISTNET, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class Model:
    def __init__(self,net,cost,optimist):
        self.net=net
        self.cost=self.create_cost(cost)
        self.optimizer=self.create_optimizer(optimist)
        pass

    def create_cost(self,cost):
        support_cost={'CROSS_ENTROPY':nn.CrossEntropyLoss(),'MSE':nn.MSELoss()}
        return support_cost[cost]

    def create_optimizer(self,optimist,**rests):
        support_optim={
            'SGD':optim.SGD(self.net.parameters(),lr=0.1,**rests),
            'ADAM':optim.Adam(self.net.parameters(),lr=0.01,**rests),
            'RMSP':optim.RMSprop(self.net.parameters(),lr=0.001,**rests)
        }
        return support_optim[optimist]

    def train(self,train_loader,epoches=3):
        for epoch in range(epoches):
            running_loss=0.0
            for i,data in enumerate(train_loader,0):
                inputs,labels = data
                inputs=inputs.cuda()
                labels=labels.cuda()
                self.optimizer.zero_grad()
                # forward +backward +optimize
                outputs = self.net(inputs)
                loss=self.cost(outputs,labels)
                loss.backward()  # 反向传播
                self.optimizer.step()

                running_loss=loss.item()
                if i % 100 == 0 :
                    print('[epoch %d,%.2f%%] loss: %.3f'%
                          (epoch+1,(i+1)*1./len(train_loader),running_loss/100))
                    running_loss = 0.0
        print('Finished Training')

    def evaluate(self,test_loader):
        print('Evaluating...')
        correct=0
        total=0
        with torch.no_grad():
            for data in test_loader:
                images,labels = data
                images=images.cuda()
                labels=labels.cuda()
                outputs = self.net(images)
                predicted = torch.argmax(outputs,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct/total))

def minist_load_data():
    transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,],[1,])]
    )

    trainset= torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=2)
    testset=torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
    testloader=torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,num_workers=2)
    return trainloader,testloader

def testData(data):
    data1=list(enumerate(data,0))
    print(data1[0])
    for i,data2 in enumerate(data,0):
        print(i,"\n",data2)
        images,labels=data2
        print(images, "\n")
        print("__________________________________________")
        print(labels)
        break

if __name__=='__main__':
    device_type = "GPU" if torch.cuda.is_available() else "CPU"  # GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    net=MNISTNET()
    net.to(device)  # 调用GPU
    model = Model(net,'CROSS_ENTROPY','RMSP')
    train_loader,test_loader = minist_load_data()
    print(train_loader)
  # testData(train_loader)
    model.train(train_loader)
    model.evaluate(test_loader)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"