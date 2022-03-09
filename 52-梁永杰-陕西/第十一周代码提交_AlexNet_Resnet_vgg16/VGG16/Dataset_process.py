import os

photos = os.listdir('./data/image/train')  # 得到目录路径中的文件和文件夹列表

with open('data/dataset.txt','w') as f:
    for photo in photos:
        name = photo.split('.')[0]    # 字符串截取   返回一个字符串列表
        if name == 'cat':
            f.write(photo + ';0\n')
        elif name == 'dog':
            f.write(photo + ';1\n')  # 若为狗 则在后方加上标签1，因为是文档所以进行换行

