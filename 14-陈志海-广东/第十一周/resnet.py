import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU, BatchNorm2d, Linear, Dropout, Softmax, AdaptiveAvgPool2d
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, k, stride):
        super(ConvBlock, self).__init__()
        f1, f2, f3 = filters
        self.bottle_net = nn.Sequential(
            Conv2d(in_channels, f1, (1, 1), stride=(stride, stride), padding=0),
            BatchNorm2d(f1),
            ReLU(True),
            Conv2d(f1, f2, (k, k), stride=(1, 1), padding=1),
            BatchNorm2d(f2),
            ReLU(True),
            Conv2d(f2, f3, (k, k), stride=(1, 1), padding=1),
            BatchNorm2d(f3))
        self.shortcut = nn.Sequential(
            Conv2d(in_channels, f3, (1, 1), stride=(stride, stride), padding=0),
            BatchNorm2d(f3))
        self._relu = ReLU(True)

    def forward(self, x):
        out_shortcut = self.shortcut(x)
        out = self.bottle_net(x)
        out += out_shortcut
        out = self._relu(out)
        return out


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, k):
        super(IdentityBlock, self).__init__()
        f1, f2, f3 = filters
        self.bottle_net = nn.Sequential(
            Conv2d(in_channels, f1, (1, 1), stride=(1, 1), padding=0),
            BatchNorm2d(f1),
            ReLU(True),
            Conv2d(f1, f2, (k, k), stride=(1, 1), padding=1),
            BatchNorm2d(f2),
            ReLU(True),
            Conv2d(f2, f3, (k, k), stride=(1, 1), padding=1),
            BatchNorm2d(f3))
        self._relu = ReLU(True)

    def forward(self, x):
        out_shortcut = x
        out = self.bottle_net(x)
        out += out_shortcut
        out = self._relu(out)
        return out


class Resnet50(nn.Module):
    def __init__(self, num_class):
        super(Resnet50, self).__init__()
        self.stage0 = nn.Sequential(
            Conv2d(3, 64, (7, 7), stride=(2, 2), padding=3),
            BatchNorm2d(64),
            ReLU(True),
            MaxPool2d((3, 3), stride=2, padding=1)
        )
        self.stage1 = nn.Sequential(
            ConvBlock(64, (64, 64, 256), 3, stride=1),
            IdentityBlock(256, (64, 64, 256), 3),
            IdentityBlock(256, (64, 64, 256), 3)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(256, (128, 128, 512), 3, stride=2),
            IdentityBlock(512, (128, 128, 512), 3),
            IdentityBlock(512, (128, 128, 512), 3),
            IdentityBlock(512, (128, 128, 512), 3),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(512, (256, 256, 1024), 3, stride=2),
            IdentityBlock(1024, (256, 256, 1024), 3),
            IdentityBlock(1024, (256, 256, 1024), 3),
            IdentityBlock(1024, (256, 256, 1024), 3),
            IdentityBlock(1024, (256, 256, 1024), 3),
            IdentityBlock(1024, (256, 256, 1024), 3),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(1024, (512, 512, 2048), 3, stride=2),
            IdentityBlock(2048, (512, 512, 2048), 3),
            IdentityBlock(2048, (512, 512, 2048), 3),
            AdaptiveAvgPool2d((1, 1)),       # 将7x7的图像均值赤化为1x1，通道数依旧为2048
        )
        self.stage5 = nn.Sequential(

            Linear(1*1*2048, num_class),     # 因为上面的avgPool把图像压缩为1x1x2048，所以in_features是2048
            Softmax()
        )

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = out.view(out.shape[0], -1)
        out = self.stage5(out)
        return out


def data_loader():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    transforms.Resize((224, 224))])
    train_set = CIFAR10("./cifar10", train=True, transform=transform, download=True)
    test_set = CIFAR10("./cifar10", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False, num_workers=0)
    return train_loader, test_loader


class Model:
    def __init__(self, net):
        self.net = net
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if i % 10 == 0:
                    print("epoch %d, %.2f, loss: %.3f" % (epoch+1, (i+1) / len(train_loader), running_loss))

    def evaluate(self, test_loader):
        total, correct = [0, 0]
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                outputs = self.net(inputs)
                predicts = np.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicts == labels).sum().item()
        print("acurracy is %.2f%%" % (correct * 100 / total))


# main
train_loader, test_loader = data_loader()
resnet = Resnet50(1000)
print(resnet)
summary(resnet, input_size=(3, 224, 224), batch_size=-1)
sys.exit()
model = Model(Resnet50(1000))
model.train(train_loader)
model.evaluate(test_loader)
