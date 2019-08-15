import cv2
import os
import numpy as np
import matplotlib.pylab as plt

os.chdir("C:\melfaiz\eigenfaces")

from training import *

def load_images(path):

    images = []
    labels = []
    n=0
    shape = (0,0)
    for img in os.listdir(path):
        n=n+1
        # extracting the name
        name = img.split(".")
        labels.append(name[0])
        #shape of image
        img = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

        img_array = img.flatten()
        img_array = np.array(img_array)
        images.append(img_array)

        shape = img.shape

    images = np.array(images)

    return images,labels,shape


def reshape_array(image,shape):

    image = image.reshape(shape)
    return image


def plot_cust(image,title,image2,title2):
    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(image,cmap='gray')
    plt.title("Name :"+title)

    plt.subplot(1,2,2)
    plt.imshow(image2,cmap='gray')
    plt.title("Predicted :"+title2)

    plt.show()

def mean(data):
    n,m = len(data),len(data[0]) #64,2490053
    mean = np.arange(m)
    for i in range(m):
        sum=0
        for j in range(n):
            sum = sum + data[j][i]
        mean[i] = sum / n
    return mean

def pca(data):

    data = data - mean(data)

    data = np.transpose(data)

    eigenvectors , eigenvalues , variance = np.linalg.svd(data, full_matrices = False )

    eigenvectors = np.transpose(eigenvectors)

    return eigenvectors, eigenvalues

def projection(sample,base):
    return np.dot(base,sample)

def img_error(a,b,E):
    err = 0
    a,b = projection(a,E) , projection(b,E)
    for i in range(E.shape[0]):
        err = err + abs( a[i] - b[i] )
    return err


def max_error(train):
    E,V = pca(train)
    err=0
    for img in train:
        for img2 in train:
            if img_error(img,img2,E) > 0 and img_error(img,img2,E) > err :
                err = img_error(img,img2,E)

    return err

[test,ts_labels,shape] = load_images("C:\melfaiz\eigenfaces\\faces\\test")
[train,tr_labels,shape] = load_images("C:\melfaiz\eigenfaces\\faces\\train")

max_error = 27000

# name=""
# no = ts_labels.index(name)
# pers = test[no]
#
#
# ERR = []
# for arr in train:
#     ERR.append(img_error(pers,arr,E))
# ERR = np.array(ERR)
#
#
# print("The error rate is : "+ str( min(ERR) ))
# true_img = reshape_array(test[no],shape)
# true_name = ts_labels[no]
#
# pred_img = reshape_array(train[np.argmin(ERR)],shape)
# pred_name= tr_labels[np.argmin(ERR)]
#
# plot_cust(true_img,true_name,pred_img,pred_name)


# plt.figure()
# plt.plot(ERR)
# plt.title("Representation of the error rate in the dataset ")
# plt.show()


def image_predict(image,train,pca,labels):

    ERR =  []

    for arr in train:
        ERR.append(img_error(image,arr,pca))
    ERR = np.array(ERR)

    pred_name= labels[np.argmin(ERR)]

    pred_name = pred_name.split(" ")

    return pred_name[0] , img_error(image, train[np.argmin(ERR)] ,pca)













