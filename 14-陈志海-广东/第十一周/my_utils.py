from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_MNIST(file_path="./dataset/mnist", transform=transforms.Compose([transforms.ToTensor,
                                             transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                             transforms.Resize((224, 224))])):
    train_set = MNIST(file_path, train=True, transform=transform, download=True)
    test_set = MNIST(file_path, train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

    return train_loader, test_loader


def load_CIFAR10(file_path="./dataset/cifar10", transform=transforms.Compose([transforms.ToTensor,
                                             transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                             transforms.Resize((224, 224))])):
    train_set = CIFAR10(file_path, train=True, transform=transform, download=True)
    test_set = CIFAR10(file_path, train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

    return train_loader, test_loader
