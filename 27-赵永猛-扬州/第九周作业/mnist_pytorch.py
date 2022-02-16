import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms

##数据集加载
def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=0)
    return trainloader, testloader


###定义网络#############
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.linear1 = nn.Linear(28*28, 512)
#         self.activation1 = nn.ReLU()
#         self.linear2 = nn.Linear(512, 512)
#         self.activation2 = nn.ReLU()
#         self.linear3 = nn.Linear(512, 10)
#         self.activation3 = nn.Softmax()
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.activation1(x)
#         x = self.linear2(x)
#         x = self.activation2(x)
#         x = self.linear3(x)
#         y = self.activation3(x)
#         return y
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

####################


net = MnistNet()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(),lr = 0.01)
train_loader, test_loader = mnist_load_data()

epochs = 3
log_step_freq = 100
loss_sum = 0.0
for step, (features, labels) in enumerate(train_loader, 1):
    # 梯度清零
    optimizer.zero_grad()

    # 正向传播求损失
    predictions = net(features)
    loss = loss_func(predictions, labels)

    # 反向传播求梯度
    loss.backward()
    optimizer.step()
    loss_sum += loss.item()
    if step % log_step_freq == 0:
        print("[step = %d] loss: %.3f" % (step, loss_sum / step))

    correct = 0
    total = 0
    with torch.no_grad():  # no grad when test and predict
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

