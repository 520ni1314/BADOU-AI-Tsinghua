import torch
import  torch.nn as nn

# 方法一：
# Sequential() 按顺序的容器
model1 = nn.Sequential()
model1.add_module("fc1",nn.Linear(3,4))
model1.add_module("fc2",nn.Linear(4,2))
model1.add_module("output",nn.Softmax(2))
# key 是自定义的
print(model1)

# 方法二：
model2 = nn.Sequential(
    nn.Conv2d(1,20,5),
    nn.ReLU(),
    nn.Conv2d(20,64,5),
    nn.ReLU()
)
# key 是从0 开始的数字
print(model2)
# 方法3
# modelList 用于存储层
model3 = nn.ModuleList([nn.Linear(3,4),nn.ReLU(),nn.Linear(4,2)])
# key 也是从零开始，单很简单
print(model3)