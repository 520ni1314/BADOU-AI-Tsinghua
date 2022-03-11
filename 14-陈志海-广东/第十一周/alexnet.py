import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.nn import Conv2d, MaxPool2d, Dropout, ReLU, Softmax, BatchNorm2d, Linear

from my_utils import load_CIFAR10

def conv_block(in_channels, out_channels, k_size, stride, padding):
    net = nn.Sequential(
        Conv2d(in_channels, out_channels, k_size, stride, padding),
        BatchNorm2d(out_channels),
        MaxPool2d((3, 3), 2),
        ReLU(True),
    )

    return net


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            conv_block(3, 96, 11, 4, 2),  # [b, 96, 55, 55]
            conv_block(96, 256, 5, 1, 2),
            Conv2d(256, 384, 3, 1, 1),
            Conv2d(384, 384, 3, 1, 1),
            conv_block(384, 256, 3, 1, 1),
        )
        self.classifier = nn.Sequential(
            Linear(6*6*256, 9216),
            ReLU(inplace=True),
            Linear(9216, 4096),
            ReLU(inplace=True),
            Linear(4096, 10),
            ReLU(inplace=True),
            Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return output


class Net():
    def __init__(self, net):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), momentum=0.9, lr=0.0001)

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print("epoch: %d, batch: %.3f, loss: %.4f" % (epoch, i/len(train_loader), loss.item()))

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            with torch.no_grad():
                outputs = self.net(inputs)
                predicts = outputs.argmax(outputs, 1)
                total += len(labels)
                correct += (predicts == labels).sum().item()
                print("accuracy: %.3f" % (correct/total))


if __name__ == "__main__":
    # file_path = "D://dataset/cifar10"
    file_path = "./dataset/cifar10"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                    transforms.Resize((224, 224))])
    train_loader, test_loader = load_CIFAR10(file_path, transform)
    epoch = 5
    alexNet = AlexNet()
    net = Net(alexNet)
    net.train(train_loader, epoch)
    net.evaluate(test_loader)
    # summary(alexNet, (3, 224, 224), 1)
