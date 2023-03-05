
from skimage import io
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt

def datainput():  #拿到所有的数据图片
    data = []
    label=[]
    path='FaceDB_orl'
    path_list = os.listdir(path)  #设置路径
    for filename in path_list:
        str= os.path.join(path, filename)
        str=str+'\*.png'
        # print(str)
        pictures = io.ImageCollection(str)  #拿到每个文件夹下面的图片
        print(pictures)
        for i in range(len(pictures)):
            data.append(np.ravel(pictures[i].reshape((1, pictures[i].shape[0] * pictures[i].shape[1]))))
            onehot=np.zeros(44)
            onehot[int(filename)-1]=(i+1)
            # print(onehot)
            label.append(onehot)
    return np.matrix(data),np.matrix(label)


def FPCA(face):

    data, label = datainput()
    X_train, y_train = sklearn.utils.shuffle(data, label) #随机打乱数据
    labely=[np.argmax(item) for item in y_train] #从onehot中制作标签
    labelx=[np.max(item) for item in y_train]
    pca = PCA(100, True, True)  #设置PCA
    fpca=pca.fit_transform(X_train) #训练PCA


    data=[]
    face=np.matrix(np.ravel(face.reshape((1, face.shape[0] * face.shape[1]))))  #将原来的向量
    test=pca.transform(face)
    from sklearn.metrics.pairwise import cosine_similarity
    ac=0
    for j in range(test.shape[0]):
        sim=[]
        csim=[]
        fsim=[]
        dict={}
        print(j)
        for num in range(fpca.shape[0]):#400

            sim.append(cosine_similarity(test[j].reshape(1,-1),fpca[num].reshape(1,-1)))
            if cosine_similarity(test[j].reshape(1,-1),fpca[num].reshape(1,-1)) > -1 :
                csim.append(labely[num]+1)
                fsim.append(labelx[num])

        sim,csim, fsim = zip(*sorted(zip(sim,csim, fsim),reverse=True))
    return csim,fsim



