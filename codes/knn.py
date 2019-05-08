from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def accuracy(predicts,labels):
    count=0
    for i in range(predicts.shape[0]):
        if predicts[i]==labels[i]:
            count+=1
    return count/predicts.shape[0]

iris_x_train=np.load('000.npy')
iris_y_train=np.load('train_labels.npy')
print(np.shape(iris_x_train))
print(np.shape(iris_y_train))

knn = KNeighborsClassifier(10)
knn.fit(iris_x_train, iris_y_train[0:55000])

iris_x_test=np.load('009.npy')
print(np.shape(iris_x_test))
iris_y_predict = knn.predict(iris_x_test)
print(np.shape(iris_y_predict))
print(iris_y_predict[0:100])

real_label=np.load('test_labels.npy')
print(real_label[0:100])

acc=accuracy(iris_y_predict,real_label)
print(acc)

'''
iris_x_train=np.asarray([[1,1,1],
              [0.5,0.5,0.2],
              [-1,-1,-1],
              [-0.5,-0.5,-0.2]], dtype=np.float32)
iris_y_train=np.asarray([0,0,1,1], dtype=np.float32)
knn = KNeighborsClassifier(2)
knn.fit(iris_x_train, iris_y_train)


iris_x_test=np.asarray([[1,1,3],[0.5,0.5,0.3],[-0.5,-0.5,-0.5]], dtype=np.float32)
print(np.shape(iris_x_test))
iris_y_predict = knn.predict(iris_x_test)
print(iris_y_predict)
print(iris_y_predict[-1])


def accuracy(predicts,labels):
    count=0
    for i in range(predicts.shape[0]):
        if predicts[i]==labels[i]:
            count+=1
    return count/predicts.shape[0]

a=np.asarray([1,2,3,4,5], dtype=np.float32)
print(a.shape[0])
b=np.asarray([1,2,3,0,5], dtype=np.float32)
acc=accuracy(a,b)
print(acc)
'''


