import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np


def load_mnist_data(batch_size):
    # 加载数据集，形成一批一批的数据
    train_data = datasets.MNIST(
        root='dataset',
        train=True,
        transform=ToTensor(),
        download=True
    )

    test_data = datasets.MNIST(
        root='dataset',
        train=False,
        transform=ToTensor(),
        download=True
    )

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    return train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_block = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_block(x)
        return logits


def train(model, dataloader, loss_function, optimizer, device):
    # 设置训练模式。如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()
    model.train()
    # 迭代训练
    for batch, (X, y) in enumerate(dataloader):
        # 将数据送入GPU或者CPU中
        X = X.to(device)
        y = y.to(device)

        # 前向传播，计算损失函数
        y_pred = model(X)
        loss = loss_function(y_pred, y)

        # 反向传播，更新权值
        optimizer.zero_grad()   # 清空上一个batch的梯度，置零
        loss.backward()  # loss反向传播，计算梯度
        optimizer.step()    # 更新权值

        # 打印信息
        size = len(dataloader.dataset)
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(model, dataloader, loss_function, device):
    # 设置测试模式。如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()
    model.eval()
    # 变量初始化为0
    test_loss = 0
    correct = 0
    with torch.no_grad():   # 强制之后的内容不进行计算图构建
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)

            test_loss += loss_function(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    PredictMode = False
    ModelSavePath = "model.pth"

    if PredictMode == False:
        # 使用mnist数据集，新训练分类模型
        BatchSize = 64
        Epochs = 5

        # 获取GPU或者CPU用于训练
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")

        model = NeuralNetwork().to(device)
        print(model)

        # 加载数据集，形成一批一批的数据
        train_data_loader, test_data_loader = load_mnist_data(BatchSize)

        # 指定损失函数和优化器
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

        # 迭代训练和验证
        for epoch in range(Epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train(model, train_data_loader, loss_func, optimizer, device)
            test(model, test_data_loader, loss_func, device)

        print("Done!")

        # 保存模型
        torch.save(model.state_dict(), ModelSavePath)
        print("Saved PyTorch Model State to model.pth")

    else:
        # 加载本地保存的模型，使用单张图进行预测
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        test_data = datasets.MNIST(
            root='dataset',
            train=False,
            transform=ToTensor(),
            download=True
        )

        model2 = NeuralNetwork()
        model2.load_state_dict(torch.load(ModelSavePath))
        model2.eval()

        # 从测试集中随机选取一个样本进行预测
        idx = np.random.randint(0, len(test_data))
        x, y = test_data[idx][0], test_data[idx][1]

        with torch.no_grad():
            pred = model2(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
