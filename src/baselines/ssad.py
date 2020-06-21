import json
import logging
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
from .shallow_ssad.ssad_convex import ConvexSSAD
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_kernels
from base.base_dataset import BaseADDataset
from networks.main import build_autoencoder
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
#from my_mnist import My_dataset
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


class My_dataset():
    def __init__(self):
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
        #anomaly_train = get_anomaly(anomaly_train)
        #anomaly_test = get_anomaly(anomaly_test)

        anomaly_train = anomaly_data_add[:1000,:]
        anomaly_test = anomaly_data_add[-1000:,:]

        train_data = np.concatenate((normal_train,anomaly_train),axis = 0)
        train_label = np.array([1]*(len(normal_train)) + [0]*len(anomaly_train))
        test_label = np.array([1]*(len(normal_test)) + [0]*len(anomaly_test)) # 1: normal 0: anomaly
        test_data = np.concatenate((normal_test,anomaly_test),axis = 0)

        self.train_loader = torch.utils.data.DataLoader(MyData(train_data,train_label),batch_size=128)
        #train_loader_init = torch.utils.data.DataLoader(MyData(train_data,train_label),batch_size=len(train_data))

        self.test_loader = torch.utils.data.DataLoader(MyData(test_data,test_label), batch_size =128)

#-----------------------------------

class SSAD(object):
    """
    A class for kernel SSAD models as described in Goernitz et al., Towards Supervised Anomaly Detection, JAIR, 2013.
    """

    def __init__(self, kernel='rbf', kappa=1.0, Cp=1.0, Cu=1.0, Cn=1.0, hybrid=False):
        """Init SSAD instance."""
        self.kernel = kernel
        self.kappa = kappa
        self.Cp = Cp
        self.Cu = Cu
        self.Cn = Cn
        self.rho = None
        self.gamma = None

        self.model = None
        self.X_svs = None

        self.hybrid = hybrid
        self.ae_net = None  # autoencoder network for the case of a hybrid model
        self.linear_model = None  # also init a model with linear kernel if hybrid approach
        self.linear_X_svs = None

        self.results = {
            'train_time': None,
            'test_time': None,
            'test_auc': None,
            'test_scores': None,
            'train_time_linear': None,
            'test_time_linear': None,
            'test_auc_linear': None
        }

        self.my_dataset = My_dataset()

    def train(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Trains the SSAD model on the training data."""
        logger = logging.getLogger()

        # do not drop last batch for non-SGD optimization shallow_ssad
        #train_loader = DataLoader(dataset=dataset.train_set, batch_size=128, shuffle=True,num_workers=n_jobs_dataloader, drop_last=False)
        train_loader = self.my_dataset.test_loader
        # Get data from loader
        X = ()
        semi_targets = []
        for data in train_loader:
            inputs, _, semi_targets_batch, _ = data
            inputs, semi_targets_batch = inputs.to(device), semi_targets_batch.to(device)
            if self.hybrid:
                inputs = self.ae_net.encoder(inputs)  # in hybrid approach, take code representation of AE as features
            X_batch = inputs.view(inputs.size(0), -1)  # X_batch.shape = (batch_size, n_channels * height * width)
            X += (X_batch.cpu().data.numpy(),)
            semi_targets += semi_targets_batch.cpu().data.numpy().astype(np.int).tolist()
        X, semi_targets = np.concatenate(X), np.array(semi_targets)

        # Training
        logger.info('Starting training...')

        # Select model via hold-out test set of 1000 samples
        gammas = np.logspace(-7, 2, num=10, base=2)
        best_auc = 0.0

        # Sample hold-out set from test set
        #_, test_loader = dataset.loaders(batch_size=128, num_workers=n_jobs_dataloader)
        #---------------------------------------------
        test_loader = self.my_dataset.test_loader


        X_test = ()
        labels = []
        for data in test_loader:
            inputs, label_batch, _, _ = data
            inputs, label_batch = inputs.to(device), label_batch.to(device)
            if self.hybrid:
                inputs = self.ae_net.encoder(inputs)  # in hybrid approach, take code representation of AE as features
            X_batch = inputs.view(inputs.size(0), -1)  # X_batch.shape = (batch_size, n_channels * height * width)
            X_test += (X_batch.cpu().data.numpy(),)
            labels += label_batch.cpu().data.numpy().astype(np.int64).tolist()
        X_test, labels = np.concatenate(X_test), np.array(labels)
        n_test, n_normal, n_outlier = len(X_test), np.sum(labels == 0), np.sum(labels == 1)
        n_val = int(0.1 * n_test)
        n_val_normal, n_val_outlier = int(n_val * (n_normal/n_test)), int(n_val * (n_outlier/n_test))
        perm = np.random.permutation(n_test)
        X_val = np.concatenate((X_test[perm][labels[perm] == 0][:n_val_normal],
                                X_test[perm][labels[perm] == 1][:n_val_outlier]))
        labels = np.array([0] * n_val_normal + [1] * n_val_outlier)

        i = 1
        for gamma in gammas:

            # Build the training kernel
            kernel = pairwise_kernels(X, X, metric=self.kernel, gamma=gamma)

            # Model candidate
            model = ConvexSSAD(kernel, semi_targets, Cp=self.Cp, Cu=self.Cu, Cn=self.Cn)

            # Train
            start_time = time.time()
            model.fit()
            train_time = time.time() - start_time

            # Test on small hold-out set from test set
            kernel_val = pairwise_kernels(X_val, X[model.svs, :], metric=self.kernel, gamma=gamma)
            scores = (-1.0) * model.apply(kernel_val)
            scores = scores.flatten()

            # Compute AUC
            auc = roc_auc_score(labels, scores)

            logger.info(f'  | Model {i:02}/{len(gammas):02} | Gamma: {gamma:.8f} | Train Time: {train_time:.3f}s '
                        f'| Val AUC: {100. * auc:.2f} |')

            if auc > best_auc:
                best_auc = auc
                self.model = model
                self.gamma = gamma
                self.results['train_time'] = train_time

            i += 1

        # Get support vectors for testing
        self.X_svs = X[self.model.svs, :]

        # If hybrid, also train a model with linear kernel
        if self.hybrid:
            linear_kernel = pairwise_kernels(X, X, metric='linear')
            self.linear_model = ConvexSSAD(linear_kernel, semi_targets, Cp=self.Cp, Cu=self.Cu, Cn=self.Cn)
            start_time = time.time()
            self.linear_model.fit()
            train_time = time.time() - start_time
            self.results['train_time_linear'] = train_time
            self.linear_X_svs = X[self.linear_model.svs, :]

        logger.info(f'Best Model: | Gamma: {self.gamma:.8f} | AUC: {100. * best_auc:.2f}')
        logger.info('Training Time: {:.3f}s'.format(self.results['train_time']))
        logger.info('Finished training.')

    def test(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Tests the SSAD model on the test data."""
        logger = logging.getLogger()

        #_, test_loader = dataset.loaders(batch_size=128, num_workers=n_jobs_dataloader)
        test_loader = self.my_dataset.test_loader
        # Get data from loader
        idx_label_score = []
        X = ()
        idxs = []
        labels = []
        for data in test_loader:
            inputs, label_batch, _, idx = data
            inputs, label_batch, idx = inputs.to(device), label_batch.to(device), idx.to(device)
            if self.hybrid:
                inputs = self.ae_net.encoder(inputs)  # in hybrid approach, take code representation of AE as features
            X_batch = inputs.view(inputs.size(0), -1)  # X_batch.shape = (batch_size, n_channels * height * width)
            X += (X_batch.cpu().data.numpy(),)
            idxs += idx.cpu().data.numpy().astype(np.int64).tolist()
            labels += label_batch.cpu().data.numpy().astype(np.int64).tolist()
        X = np.concatenate(X)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()

        # Build kernel
        kernel = pairwise_kernels(X, self.X_svs, metric=self.kernel, gamma=self.gamma)

        scores = (-1.0) * self.model.apply(kernel)

        self.results['test_time'] = time.time() - start_time
        scores = scores.flatten()
        self.rho = -self.model.threshold

        # Save triples of (idx, label, score) in a list
        idx_label_score += list(zip(idxs, labels, scores.tolist()))
        self.results['test_scores'] = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.results['test_auc'] = roc_auc_score(labels, scores)

        # If hybrid, also test model with linear kernel
        if self.hybrid:
            start_time = time.time()
            linear_kernel = pairwise_kernels(X, self.linear_X_svs, metric='linear')
            scores_linear = (-1.0) * self.linear_model.apply(linear_kernel)
            self.results['test_time_linear'] = time.time() - start_time
            scores_linear = scores_linear.flatten()
            self.results['test_auc_linear'] = roc_auc_score(labels, scores_linear)
            logger.info('Test AUC linear model: {:.2f}%'.format(100. * self.results['test_auc_linear']))
            logger.info('Test Time linear model: {:.3f}s'.format(self.results['test_time_linear']))

        # Log results
        logger.info('Test AUC: {:.2f}%'.format(100. * self.results['test_auc']))
        logger.info('Test Time: {:.3f}s'.format(self.results['test_time']))
        logger.info('Finished testing.')

    def load_ae(self, dataset_name, model_path):
        """Load pretrained autoencoder from model_path for feature extraction in a hybrid SSAD model."""

        model_dict = torch.load(model_path, map_location='cpu')
        ae_net_dict = model_dict['ae_net_dict']
        if dataset_name in ['mnist', 'fmnist', 'cifar10']:
            net_name = dataset_name + '_LeNet'
        else:
            net_name = dataset_name + '_mlp'

        if self.ae_net is None:
            self.ae_net = build_autoencoder(net_name)

        # update keys (since there was a change in network definition)
        ae_keys = list(self.ae_net.state_dict().keys())
        for i in range(len(ae_net_dict)):
            k, v = ae_net_dict.popitem(False)
            new_key = ae_keys[i]
            ae_net_dict[new_key] = v
            i += 1

        self.ae_net.load_state_dict(ae_net_dict)
        self.ae_net.eval()

    def save_model(self, export_path):
        """Save SSAD model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load SSAD model from import_path."""
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
