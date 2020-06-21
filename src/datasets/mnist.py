from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import random

import numpy as np

class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        '''
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)
        '''
        self.normal_classes = tuple([0,1,2,3,4,5,6,7,8,9])
        self.outlier_classes = []


        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        #target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        target_transform = None

        
        # Get train set
        train_set = MyMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform,
                            download=True)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        print("semi_targets",len(semi_targets))
        print("idx",len(idx))
        #train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                download=True)


class MyMNIST(MNIST):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)
        self.anomaly_rate = 0.05
        self.semi_label_rate = 0.01

        def get_anomaly(anomaly_data):
            n_anomaly = len(anomaly_data)
            dim = 28
            #print("anomaly_data",anomaly_data.shape)
            a1,a2 = anomaly_data[:n_anomaly//2,:dim//2,:],anomaly_data[:n_anomaly//2,dim//2:,:]
            b1,b2 = anomaly_data[n_anomaly//2:,:dim//2,:],anomaly_data[n_anomaly//2:,dim//2:,:]

            #print("a1",a1.shape)
            #print("b2",b2.shape)
            anomaly_data1 = np.concatenate((a1,b2),axis = 1)
            anomaly_data2 = np.concatenate((b1,a2),axis = 1)
            anomaly_data = np.concatenate((anomaly_data1,anomaly_data2),axis = 0)
            return anomaly_data

        if not self.train:
            #pass
            test_data_normal = self.test_data[:9000,:,:]
            test_data_anomaly = get_anomaly(self.test_data[9000:,:,:])

            data = np.concatenate((test_data_normal,test_data_anomaly),axis = 0)
            targets = np.array([0]*(len(test_data_normal)) + [1]*len(test_data_anomaly))
            #np.random.seed(1)
            #np.random.shuffle(data)
            #np.random.seed(1)
            #np.random.shuffle(targets)

            self.data = torch.from_numpy(data)
            self.targets = torch.from_numpy(targets)

        else:
            n_train = len(self.train_data)
            n_normal = n_train - int(self.anomaly_rate*n_train)
            n_anomaly = int(self.anomaly_rate*n_train)
            normal_train = self.train_data[:n_normal,:,:]
            tobe_anomaly_train = self.train_data[n_normal:,:,:]
            print("normal_train",len(normal_train))
            print("tobe_anomaly_train",len(tobe_anomaly_train))

            anomaly_train = get_anomaly(tobe_anomaly_train)
            print("anomaly_train",len(anomaly_train))

            data = np.concatenate((normal_train,anomaly_train),axis = 0)

            semi_target = np.array([0 for _ in range(n_normal-50)] + [1 for _ in range(50)] + [-1 for _ in range(50)] + [0 for _ in range(n_anomaly)])
            self.semi_target = torch.from_numpy(semi_target)
            #np.random.seed(1)
            #np.random.shuffle(data)
            #print("data",len(data))
            #print("self.data",len(self.data))
            self.data = torch.from_numpy(data)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])
        #img, target, semi_target = self.data[index], int(self.targets[index]), int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
