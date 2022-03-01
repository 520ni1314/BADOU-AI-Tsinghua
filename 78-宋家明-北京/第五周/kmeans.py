import numpy as np
import random
import cv2
import operator

def kmean_iter(data,num,meansample):
    ori_data = data
    num = num
    meansample = meansample
    data_dis = []
    for d in meansample:
        d_dis = ori_data-d
        d_dis = np.sum(d_dis**2,axis=1)
        data_dis.append(d_dis)

    data_dis = np.array(data_dis,dtype=np.int16)
    data_class = np.argmin(data_dis,axis=0)
    print('000:',data_class)
    data_class_dict = {}
    for i in range(num):
        data_class_dict[str(i)] = []
    for cid,c in enumerate(data_class):
        data_class_dict[str(c)].append(data[cid])
    for i in range(num):
        data = np.array(data_class_dict[str(i)],dtype=np.int16)
        n,d = data.shape
        data_class_dict[str(i)] = np.sum(data,axis=0)/n
    meansample = []
    for i in range(num):
        meansample.append(data_class_dict[str(i)])
    return data_class,meansample

def kmeans_fun(data,num):
    sample_num, sample_dim = data.shape[0],data.shape[1]
    samplelist = [i for i in range(sample_num)]
    randomsample = random.sample(samplelist,num)
    randomsample = [data[i] for i in randomsample]

    data_oldclass, meansample = kmean_iter(data,num,randomsample)
    while True:
        data_class, meansample = kmean_iter(data,num,meansample)
        if operator.eq(list(data_oldclass),list(data_class)):
            break
        else:
            data_oldclass = data_class
    print('dataclass:    ',data_class)
    print('dataoldclass: ',data_oldclass)


        

if __name__=='__main__':

    img = cv2.imread('../lenna.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(2,20))
    datas = img.astype(np.int16)



    n_class = 4
    kmeans_fun(datas,n_class)


