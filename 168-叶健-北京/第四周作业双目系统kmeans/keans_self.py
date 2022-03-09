from sklearn.cluster import KMeans
import random


data=[]
count=100
for i in range(count):
    if i < 40:
        data.append([random.randint(0,30),random.randint(0,30)])
    elif 40<=i < 80:
        data.append([random.randint(30,60),random.randint(30,60)])
    else:
        data.append([random.randint(80, 90), random.randint(80, 90)])
print(data)

clf=KMeans(n_clusters=3)
label=clf.fit_predict(data)
print(label)

import matplotlib.pyplot as plt
x0=[]
x1=[]
x2=[]
y0=[]
y1=[]
y2=[]
for i in range(count):
    if label[i]==0:
        x0.append(data[i][0])
        y0.append(data[i][1])
    elif label[i]==1:
        x1.append(data[i][0])
        y1.append(data[i][1])
    else:
        x2.append(data[i][0])
        y2.append(data[i][1])
plt.scatter(x0,y0,c='b',marker="p")
plt.scatter(x1,y1,c='g',marker="*")
plt.scatter(x2,y2,c='r',marker="x")
plt.legend(["A","B","C"])
plt.show()