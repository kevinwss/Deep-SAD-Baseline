import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import random
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset

import gzip
import os
from urllib.request import urlretrieve
from PIL import Image

batch_size = 200


from base.torchvision_dataset import TorchvisionDataset
#-------------------------------------------------------------------


def mnist(path=None):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing MNIST. Default is
            /home/USER/data/mnist or C:\Users\USER\data\mnist.
            Create if nonexistant. Download any missing files.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values.
            Columns of labels are a onehot encoding of the correct class.
    """
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'mnist')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download any missing files
    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        def _onehot(integer_labels):
            """Return matrix whose rows are onehot encodings of integers."""
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))

    return train_images, train_labels, test_images, test_labels


class MyData(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label
        self.to_tensor = transforms.ToTensor()
 
    def __getitem__(self, index):
        
        image = np.reshape((self.data[index]),(28,28))
        #print("1")
        #print("image",image.shape)
        #image = transforms.ToPILImage()(image)
        image = Image.fromarray(np.uint8(image))
        #print("2.5")
        img_as_tensor = self.to_tensor(image)
        #print("2")
        
        single_image_label = self.label[index]
        #print("3")
        return img_as_tensor,single_image_label, single_image_label, index
    def __len__(self):
        return len(self.data)


class MNIST_Dataset():
    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)
    #def __init__(self):
        train_data,train_label, test_data, test_label = mnist("MNIST_data")
        data0 = train_data
        data = []
        anomaly_data_add = []
        labels = train_label
        dim = 28*28
        mb_size =64
        anomaly_num = 2

        for i in range(len(labels)):
            if (labels[i]).tolist().index(1) != anomaly_num:
                data.append(data0[i])
            else:
                anomaly_data_add.append(data0[i])

        anomaly_data_add = np.array(anomaly_data_add)
        data = np.array(data)
        use_class_anonmaly = True

        #---------------------
        def get_anomaly(anomaly_data):
            n_anomaly = len(anomaly_data)
            a1,a2 = anomaly_data[:n_anomaly//2,:dim//2],anomaly_data[:n_anomaly//2,dim//2:]
            b1,b2 = anomaly_data[n_anomaly//2:,:dim//2],anomaly_data[n_anomaly//2:,dim//2:]
            print("a1",a1.shape)
            print("b2",b2.shape)

            anomaly_data1 = np.concatenate((a1,b2),axis = 1)
            anomaly_data2 = np.concatenate((b1,a2),axis = 1)
            anomaly_data = np.concatenate((anomaly_data1,anomaly_data2),axis = 0)

            return anomaly_data

        n_data = data.shape[0]
        anomaly_rate = 0.1
        train_rate = 0.8
        n_anomaly = int(n_data*anomaly_rate)
        n_train = int(n_data*train_rate)
        n_test = n_data - n_train
        seed = 1

        train_data = data[:n_train,:]
        test_data = data[n_train:,:]

        normal_train = train_data[:n_train-int(anomaly_rate*n_train),:]
        normal_test = test_data[:n_test-int(0.1*n_test),:]  
        anomaly_train = train_data[n_train-int(anomaly_rate*n_train):,:]
        anomaly_test = test_data[n_test-int(0.1*n_test):,:]


        if len(anomaly_train)%2==1:
            anomaly_train = anomaly_train[:len(anomaly_train)-1,:]
        if len(anomaly_test)%2==1:
            anomaly_test = anomaly_test[:len(anomaly_test)-1,:]
        anomaly_train = get_anomaly(anomaly_train)
        anomaly_test = get_anomaly(anomaly_test)

        #anomaly_train = anomaly_data_add[:1000,:]
        #anomaly_test = anomaly_data_add[-1000:,:]

        train_data = np.concatenate((normal_train,anomaly_train),axis = 0)
        train_label = np.array([1]*(len(normal_train)) + [0]*len(anomaly_train))
        test_label = np.array([1]*(len(normal_test)) + [0]*len(anomaly_test)) # 1: normal 0: anomaly
        test_data = np.concatenate((normal_test,anomaly_test),axis = 0)

        #self.train_loader = torch.utils.data.DataLoader(MyData(train_data,train_label),batch_size=args.batch_size)
        #self.test_loader = torch.utils.data.DataLoader(MyData(test_data,test_label),batch_size=args.batch_size)
        '''
        self.train_set = MyData(train_data,train_label)
        self.test_set = MyData(test_data,test_label)
        '''
        self.train_loader = torch.utils.data.DataLoader(MyData(train_data,train_label),batch_size=batch_size)
        self.test_loader = torch.utils.data.DataLoader(MyData(test_data,test_label),batch_size=batch_size)

