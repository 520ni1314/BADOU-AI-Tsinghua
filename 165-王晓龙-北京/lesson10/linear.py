import  torch

# 创建一个线性层类，并继承nn.module
class Linear(torch.nn.Module):
    # 初始化
    # 参数： 输入特征数，输出特征数，有截距
    def __init__(self,input_features,out_featurs,bias=True):
        super(Linear,self).__init__() # 继承init

        # 随机初始化权重
        self.weight = torch.nn.Parameter(torch.randn(out_featurs,input_features))
        print(self.weight)
        # 随机初始化截距
        self.bias = torch.nn.Parameter(torch.randn(out_featurs))

    def forward(self,x): # 定义前向计算
        x = x.mm(self.weight) # y=wx
        # x = x +self.bias.expand_as(x) #expand_as(x)的size是一样大的。且是不共享内存的。
        return x  # y = wx +b

if __name__ == "__main__":
    net = Linear(3,2) # 实例化Linear ,并传入input =3 ,output =2
    t = torch.randn(3,2)
    x = net.forward(t)  # 调用 forward 方法
    print("11",x)

# 思路
# 1 .创建类(继承nn.model)
# 2. init 方法 ：初始化参数和bias
# 3. forward 前向计算
# 4.实例化类。并调用其方法