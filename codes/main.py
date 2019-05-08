import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy import signal
import cv2 as cv

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE





def loadImageSet(filename):
    binfile = open(filename, 'rb')  # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组

    return imgs, head


def loadLabelSet(filename):
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    return labels, head


if __name__ == "__main__":
    file1 = 'train-images.idx3-ubyte'

    size=60000
    imgs, data_head = loadImageSet(file1)
    print('data_head:', data_head)
    print(type(imgs))
    print('imgs_array:', imgs)
    #print(np.reshape(imgs[0, :], [28, 28]))  # 取出其中一张图片的像素，转型为28*28，大致就能从图像上看出是几啦

    file2 = 'train-labels.idx1-ubyte'

    labels, data_head = loadLabelSet(file2)
    print("real labels")
    print(labels[0:50])
    np.save('train_labels.npy', labels)





    im=np.reshape(imgs[2, :], [28, 28])
    fig = plt.figure()
    #plotwindow = fig.add_subplot(111)
    #plt.imshow(im, cmap='gray')
    #plt.show()


    picked_images = imgs[0:size, :]
    print(np.shape(picked_images))
    np.save('original_60000.npy', picked_images)


    kmeans_r = KMeans(n_clusters=10, random_state=0, precompute_distances=True).fit(picked_images)
    cluster_labels = kmeans_r.labels_
    np.save('labels_kmean_60000.npy', cluster_labels)


    
    list_index = []
    cluster_labels = kmeans_r.labels_
    cluster_labels = cluster_labels.tolist()

    for i in range(size):
        ind = np.random.randint(10, size - 10)
        aim_labels = cluster_labels[ind - 10:ind + 10]
        curr_label = cluster_labels[i]

        cou = aim_labels.count(curr_label)

        if cou == 0:
            list_index.append(i)
        elif cou > 0:
            fi_ind = aim_labels.index(curr_label)
            real_ind = ind - 10 + fi_ind
            list_index.append(real_ind)
        else:
            print("err")

    aim_cluster = np.empty((0, 28 * 28))
    for num in range(len(list_index)):
        aim_cluster = np.vstack((aim_cluster, picked_images[list_index[num], :]))

    np.save('aim_cluster_kmean_60000.npy', aim_cluster)



    list_index_2 = []
    real_labels = labels[0:size]
    real_labels = real_labels.tolist()

    for i in range(size):
        ind = np.random.randint(10, size - 10)
        aim_labels = real_labels[ind - 10:ind + 10]
        curr_label = real_labels[i]

        cou = aim_labels.count(curr_label)

        if cou == 0:
            list_index_2.append(i)
        elif cou > 0:
            fi_ind = aim_labels.index(curr_label)
            real_ind = ind - 10 + fi_ind
            list_index_2.append(real_ind)
        else:
            print("err")

    aim_cluster_real = np.empty((0, 28 * 28))
    for num in range(len(list_index_2)):
        aim_cluster_real = np.vstack((aim_cluster_real, picked_images[list_index_2[num], :]))

    np.save('aim_cluster_real_60000.npy', aim_cluster_real)

    '''
    distance=np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            img = np.reshape(imgs[i, :], [28, 28])
            img.dtype = np.uint8
            temp = np.reshape(imgs[j, :], [28, 28])
            temp.dtype = np.uint8
            result = cv.matchTemplate(img, temp, cv.TM_CCOEFF_NORMED)
            distance[i,j]=result
    '''

'''
    kmeans_r = KMeans(n_clusters=10, random_state=0, precompute_distances=True).fit(imgs[0:size, :])
    label_r = kmeans_r.labels_[0:50]
    print("result of KMeans R")
    print(label_r)

    list0 = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []

    list0_la = []
    list1_la = []
    list2_la = []
    list3_la = []
    list4_la = []
    list5_la = []
    list6_la = []
    list7_la = []
    list8_la = []
    list9_la = []



    for num in range(size):
        if kmeans_r.labels_[num] == 0:
            list0.append(num)
            list0_la.append(labels[num])
        elif kmeans_r.labels_[num] == 1:
            list1.append(num)
            list1_la.append(labels[num])
        elif kmeans_r.labels_[num] == 2:
            list2.append(num)
            list2_la.append(labels[num])
        elif kmeans_r.labels_[num] == 3:
            list3.append(num)
            list3_la.append(labels[num])
        elif kmeans_r.labels_[num] == 4:
            list4.append(num)
            list4_la.append(labels[num])
        elif kmeans_r.labels_[num] == 5:
            list5.append(num)
            list5_la.append(labels[num])
        elif kmeans_r.labels_[num] == 6:
            list6.append(num)
            list6_la.append(labels[num])
        elif kmeans_r.labels_[num] == 7:
            list7.append(num)
            list7_la.append(labels[num])
        elif kmeans_r.labels_[num] == 8:
            list8.append(num)
            list8_la.append(labels[num])
        elif kmeans_r.labels_[num] == 9:
            list9.append(num)
            list9_la.append(labels[num])
        else:
            print("err")

    aim_list=[]

    if (float(list0_la.count(6))/len(list0_la))>0.8:
        aim_list=list0
    elif (float(list1_la.count(6))/len(list1_la))>0.8:
        aim_list=list1
    elif (float(list2_la.count(6))/len(list2_la))>0.8:
        aim_list=list2
    elif (float(list3_la.count(6))/len(list3_la))>0.8:
        aim_list=list3
    elif (float(list4_la.count(6))/len(list4_la))>0.8:
        aim_list=list4
    elif (float(list5_la.count(6))/len(list5_la))>0.8:
        aim_list=list5
    elif (float(list6_la.count(6))/len(list6_la))>0.8:
        aim_list=list6
    elif (float(list7_la.count(6))/len(list7_la))>0.8:
        aim_list=list7
    elif (float(list8_la.count(6))/len(list8_la))>0.8:
        aim_list=list8
    elif (float(list9_la.count(6))/len(list9_la))>0.8:
        aim_list=list9
    else:
        print("err")

    aim_cluster=np.empty((0,28*28))
    for num in range(len(aim_list)):
        aim_cluster = np.vstack((aim_cluster, picked_images[aim_list[num], :]))

    aim_list_index = np.array(aim_list)
    np.save('aim_list_index_40000.npy', aim_list_index)
    np.save('aim_cluster_40000.npy', aim_cluster)
'''


'''
    #draw result pick 6
    X=imgs[0:size, :]
    X = TSNE(learning_rate=100).fit_transform(X)
    y=labels[0:size]
    cents = kmeans_r.cluster_centers_  # 质心
    labels = kmeans_r.labels_  # 样本点被分配到的簇的索引
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
    n_clusters = 10
    for i in range(n_clusters):
        index = np.nonzero(labels == i)[0]
        x0 = X[index, 0]
        x1 = X[index, 1]
        y_i = y[index]
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(int(y_i[j])), color=colors[i], \
                     fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(cents[i, 0], cents[i, 1], marker='x', color=colors[i], linewidths=12)
    plt.axis([-50, 50, -50, 50])
    plt.show()
'''









'''
    agg = AgglomerativeClustering(n_clusters=10, affinity='precomputed',linkage='complete')
    all_labels_1=agg.fit_predict(distance)  # Returns class labels.
    label_1=all_labels_1[0:40]
    print("result of AgglomerativeClustering")
    print(label_1)


    kmeans = KMeans(n_clusters=10, random_state=0,precompute_distances=True).fit(distance)
    label_2 = kmeans.labels_[0:40]
    print("result of KMeans")
    print(label_2)

    brc = Birch(branching_factor=50, n_clusters=10, threshold=0.5,compute_labels = True)
    all_labels_3 = brc.fit_predict(distance)
    label_3=all_labels_3[0:40]
    print("result of Birch")
    print(label_3)
    
    spectralCluster = SpectralClustering(n_clusters=10, affinity='precomputed')
    all_labels_4 = spectralCluster.fit_predict(distance)  # Returns class labels.
    label_4 = all_labels_4[0:40]
    print("result of SpectralClustering")
    print(label_4)

  


    label_0=[]
    for i in range(size):
        if labels[i]==0:
            label_0.append(all_labels_4[i])
    print(label_0)

    num=0
    for i in range(np.size(label_0)):
        if label_0[i]==7:
            num+=1

    acc=float(num)/np.size(label_0)
    print(acc)
'''













