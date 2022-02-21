# -- coding:utf-8 --
import pandas as pd
sales = pd.read_csv("./train_data.csv",sep="\s*,\s*",engine="python")
X = sales["X"].values
Y = sales["Y"].values
#print(sales)
# 参数初始化
s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4
for i in range(n):
    s1 = s1 + X[i]
    s2 = s2 + Y[i]
    s3 = s3 + X[i] * X[i]
    s4 = s4 + X[i] * Y[i]
k = (n * s4 - s1 * s2)/(n * s3 - s1 * s1)
b = s2/n - k * (s1/n)
print("coeff:{},intercept:{}".format(k,b))
