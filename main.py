import cv2
import os
import numpy as np
import matplotlib.pylab as plt


def load_images(path):

    images = []
    labels = []
    n=0
    for img in os.listdir(path):
        n=n+1
        # extracting the name
        name = img.split(".")
        labels.append(name[0])
        #shape of image
        img = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        shape=img.shape
        img_array = img.flatten()
        images.append(img_array)

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
    n,len = data.shape
    mean = np.arange(len)
    for i in range(len):
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

    return eigenvectors,eigenvalues

def projection(sample,base):
    return np.dot(base,sample)

def img_error(a,b,E):
    err = 0
    a,b = projection(a,E) , projection(b,E)
    for i in range(E.shape[0]):
        err = err + abs( a[i] - b[i] )
    return err

[test,ts_labels,shape] = load_images(".\database\\train")
[train,tr_labels,shape] = load_images(".\database\\test")

E,V = pca(train)

name="zach2"
no = ts_labels.index(name)
pers = test[no]


ERR = []
for arr in train:
    ERR.append(img_error(pers,arr,E))
ERR = np.array(ERR)

true_img = reshape_array(test[no],shape)
true_name = ts_labels[no]

pred_img = reshape_array(train[np.argmin(ERR)],shape)
pred_name= tr_labels[np.argmin(ERR)]

plot_cust(true_img,true_name,pred_img,pred_name)


plt.figure()
plt.plot(ERR)
plt.show()

print("The error value is : "+ str( min(ERR)/max(ERR) ) )














