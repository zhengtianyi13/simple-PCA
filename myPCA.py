# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

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
    print(path_list)
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

#
#
# def splitdata(data,label):
#     X_train, X_test, y_train, y_test = train_test_split(
#         data, label, test_size=0.2, random_state=42)
#     return  X_train, X_test, y_train, y_test


def FPCA(face):
    print("f")
    print(face)
    data, label = datainput()
    X_train, y_train = sklearn.utils.shuffle(data, label) #随机打乱数据
    labely=[np.argmax(item) for item in y_train] #从onehot中制作标签
    labelx=[np.max(item) for item in y_train]
    print('xxxx')
    print(labelx)
    print(len(labely))
    print('yyy')
    print(y_train.shape)
    pca = PCA(100, True, True)  #设置PCA
    fpca=pca.fit_transform(X_train) #训练PCA
    # V=pca.components_
    # fig,axes=plt.subplots(10,10,figsize=(15,15),subplot_kw={"xticks":[],"yticks":[]})
    # for i,ax in enumerate(axes.flat):
    #     print("88")
    #     print(ax.imshow(V[i,:].reshape(112,92),cmap="gray"))
    #
    #
    print(fpca.shape)
    # img,labelT=Tinput()
    data=[]
    print(np.ravel(face.reshape((1, face.shape[0] * face.shape[1]))))
    face=np.matrix(np.ravel(face.reshape((1, face.shape[0] * face.shape[1]))))  #将原来的向量
    print(face.shape)
    test=pca.transform(face)
    print(test.shape)
    from sklearn.metrics.pairwise import cosine_similarity
    ac=0
    for j in range(test.shape[0]):
        sim=[]
        csim=[]
        fsim=[]
        dict={}
        print(j)
        for num in range(fpca.shape[0]):#400
            # print(test)
            # print(faceF[num])
            # print('sim++++++++++++++++')
            # print(cosine_similarity(test,faceF[num].reshape(1,-1)))
            sim.append(cosine_similarity(test[j].reshape(1,-1),fpca[num].reshape(1,-1)))
            if cosine_similarity(test[j].reshape(1,-1),fpca[num].reshape(1,-1)) > -1 :
                print('888888')
                print(labely[num]+1)
                csim.append(labely[num]+1)
                fsim.append(labelx[num])
        print('sim11')
        print(sim)
        sim,csim, fsim = zip(*sorted(zip(sim,csim, fsim),reverse=True))
        print('sim22')
        print(sim)
        print(csim)
        print(fsim)

        print(labely[sim.index(max(sim))]+1)
        print(labelx[sim.index(max(sim))])
        print(max(sim))



    return csim,fsim





# def print_hi(name):
#     # 在下面的代码行中使用断点来调试脚本。
#     # data,label=datainput()
#     # X_train, y_train=sklearn.utils.shuffle(data,label)
#     # print('X_train :')
#     # print(X_train)
#     # print('Y_train :')
#     # print(y_train)
#     # pca = PCA(100, True, True)
#     # pca.fit(X_train)
#     # Feature,labelF=Dinput()
#     # # print(Feature.shape)
#     # Fface=pca.transform(Feature)
#     # print(Fface.shape)
#
#
#     FPCA()
#
#
#
#     # print(Feature)
#     #
#     # #将所有人脸的数据取平均
#     # i=0
#     # mean=np.zeros(Feature[0].shape)
#     # print(mean.shape)
#     # faceF=[]
#     # while(i<320):
#     #     if (i+1)%8==0:
#     #         # print(i)
#     #         # print(mean)
#     #         faceF.append(mean/9)
#     #         # mean.clear()
#     #     else:
#     #         mean=mean+Feature[i]
#     #
#     #     i=i+1
#     # # print('face :')
#     # # print(faceF)
#     # # testDataS = pca.transform(X_test)
#     #
#     # # img=np.ravel(img.reshape(1, -1))
#     # test = pca.transform([0])
#     # sim=[]
#     # sim.append(test)
#     # img=io.imread('FaceDB_orl/019/10.png')
#     # img=img.reshape( 1,img.shape[0] *img.shape[1])
#
#
#
#
#     #
#     # img,labelT=Tinput()
#     # test=pca.transform(img)
#     # print(test.shape)
#     # from sklearn.metrics.pairwise import cosine_similarity
#     # ac=0
#     # for j in range(test.shape[0]):
#     #     sim=[]
#     #     for num in range(Fface.shape[0]):
#     #         # print(test)
#     #         # print(faceF[num])
#     #         # print('sim++++++++++++++++')
#     #         # print(cosine_similarity(test,faceF[num].reshape(1,-1)))
#     #         sim.append(cosine_similarity(test[j].reshape(1,-1),Fface[num].reshape(1,-1)))
#     #
#     #     print(sim)
#     #     print(sim.index(max(sim))+1)
#     #     if sim.index(max(sim))==j:
#     #         ac=ac+1
#     #
#     # print('准确率：')
#     # print(ac/test.shape[0])
#
#


    # PCA(150)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
