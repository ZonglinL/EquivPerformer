import sys
import dgl
import numpy as np
import torch
import os
import pickle
import time
import h5py
import random
from sklearn.neighbors import NearestNeighbors
DTYPE = np.float32

from .provider import random_scale_point_cloud, jitter_point_cloud,rotate_perturbation_point_cloud



def loadh5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label

def normalize_data(pcs):
    for pc in pcs:
        # get furthest point distance then normalize
        d = max(np.sum(np.abs(pc) ** 2, axis=-1) ** (1. / 2))
        pc /= d

    # pc[:,0]/=max(abs(pc[:,0]))
    # pc[:,1]/=max(abs(pc[:,1]))
    # pc[:,2]/=max(abs(pc[:,2]))

    return pcs

def far_sampling(data, n_points):
    N, P, __ = data.shape

    r = torch.zeros(N, n_points, __)
    sample_id_whole = np.zeros((N,n_points))
    for i in range(N):
        source = data[i] ## P 3
        id = np.random.randint(low=0, high=P, size=1)
        sample_id = [id.item()]

        for j in range(n_points):

            tmp = source[sample_id].reshape(len(sample_id),1,3)
            dist = ((source - tmp)**2).sum(-1) ## len(sample_id) P
            dist = dist.min(0) ## P
            far_id = np.argmax(dist).item()
            sample_id_whole[i,j] = far_id
            sample_id.append(far_id) ## append farthest point


        sample_id = np.array(sample_id) ## n_points farthest id
        sample_id = torch.tensor(sample_id).to(torch.long)

        #r[i] = torch.tensor(source)[sample_id]

    return sample_id_whole

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return (x @ Q).to(torch.float32)
class PC3DDataset(torch.utils.data.Dataset):

    node_feature_size = 1

    def __init__(self, FLAGS, split):
        """Create a dataset object"""

        # Data shapes:
        #  points :: [samples, points, 3]
        #  label  :: [samples, 1]

        self.FLAGS = FLAGS
        self.split = split
        self.n_points = FLAGS.num_points

        self.num_corners = 8
        self.num_neighbors = 36
        self.region = int(2048//(1.2*self.num_corners))





        assert split in ["test", "train"]
        if split == "train":

            filename = 'h5_files/split1_nobg/training_objectdataset.h5'
        else:
            filename = 'h5_files/split1_nobg/test_objectdataset.h5'



        # data.shape => 11481, 2048, 3
        # label.shape => 11481, 2048,
        points, label = loadh5(filename)            #data = f['data'][:]
            #label = f['label'][:]


        if self.split == 'test':
            #points = far_sampling(points, self.n_points)
            print('test points sampled')

        points, label = torch.tensor(points).to(torch.float), torch.tensor(label)
        if self.split == 'test':
            r = RandomRotation()
            points = r(points)

        data = {'points':points,
                'label':label}

        self.data = data
        self.len = data['points'].shape[0]

    def __len__(self):
        return self.len

    def connect_fully(self, num_atoms):
        """Convert to a fully connected graph"""
        # Initialize all edges: no self-edges
        src = []
        dst = []
        for i in range(num_atoms):
            src.append(i)
            dst.append(i)
        return np.array(src), np.array(dst)

    def __getitem__(self, idx):

        # select a start and a target frame



        x_0 = self.data['points'][idx] ## 2048 * 3
        if self.split == 'train':

            x_0 = x_0.unsqueeze(0).numpy()
            x_0 = jitter_point_cloud(x_0,0.005)
            #x_0 = random_scale_point_cloud(x_0,0.9,1.1)
            x_0 = torch.tensor(x_0)
            x_0 = x_0.squeeze(0).to(torch.float32)


        P, D = x_0.shape

        index = torch.LongTensor(random.sample(range(P), self.n_points))

        x_sample = x_0[index]

        label = np.zeros(self.FLAGS.num_class)
        label[self.data['label'][idx]] = 1
        label_0 = torch.tensor(label.astype(DTYPE))


        """
        Entire graph
        """
        # Create graph (connections only, no bond or feature information yet)
        indices_src, indices_dst = self.connect_fully(self.n_points)
        G = dgl.graph((indices_src, indices_dst))
        avg = x_0.mean(dim=0)
        x_sample -= avg
        ### add bond & feature information to graph
        G.ndata['x'] = torch.unsqueeze(x_sample, dim=1)  # [N, 1, 3]
        G.edata['d'] = torch.clone(torch.unsqueeze(x_sample, dim=1)).detach()
        
            
        



        return G, label_0

