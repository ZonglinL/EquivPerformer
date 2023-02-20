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
    for i in range(N):
        source = data[i]
        id = np.random.randint(low=0, high=P, size=1)
        sample_id = [id]
        for j in range(n_points-1):
            dist = [] ## iterate over number of selected points
            for k in sample_id:
                dist_k = ((source - source[k])**2).sum(-1) ## distance to kth points in selected P
                dist.append(dist_k) ## append, KxP

            dist = np.min(np.array(dist),axis = 0).squeeze()## smallest distance to sampled points length P
            sample_id.append(np.argmax(dist).reshape(1,)) ## append farthest point
        sample_id = np.concatenate(sample_id) ## n_points farthest id
        sample_id = torch.tensor(sample_id).to(torch.long)

        r[i] = torch.tensor(source)[sample_id]

    return r

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


        assert split in ["test", "train"]
        if split == "train":

            filename = 'h5_files/main_split_nobg/training_objectdataset.h5'
        else:
            filename = 'h5_files/main_split_nobg/test_objectdataset.h5'



        # data.shape => 11481, 2048, 3
        # label.shape => 11481, 2048,
        points, label = loadh5(filename)            #data = f['data'][:]
            #label = f['label'][:]


        if self.split == 'test':
            #points = far_sampling(points, self.n_points)

            print('test points sampled')


        points, label = torch.tensor(points).to(torch.float), torch.tensor(label)

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
            '''
            for j in range(num_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)
            '''
            src.append(i)
            dst.append(i)
        return np.array(src), np.array(dst)

    def __getitem__(self, idx):

        # select a start and a target frame

        if self.split == 'test':
            r  = RandomRotation()
            x_0 = r(torch.tensor(self.data['points'][idx])) #.astype(DTYPE))
        else:
            x_0 = torch.tensor(self.data['points'][idx])

        #knn = NearestNeighbors(n_jobs=-1)
        #knn.fit(x_0)

        P, D = x_0.shape
        index = torch.LongTensor(random.sample(range(P), self.n_points))
        x_sample = x_0[index]

        #dist,ind = knn.kneighbors(x_sample,11)
        #ind = ind[:,1:] ## num_points 10 3
        #features = x_0[ind].mean(1).view(self.n_points,1,D)


        '''
        if self.split == 'train':
            index = torch.LongTensor(random.sample(range(P), self.n_points))  ## random selection
            x_0 = x_0[index]
        '''

        label = np.zeros(self.FLAGS.num_class)
        label[self.data['label'][idx]] = 1
        label_0 = torch.tensor(label.astype(DTYPE))

        # Create graph (connections only, no bond or feature information yet)
        indices_src, indices_dst = self.connect_fully(self.n_points)
        G = dgl.graph((indices_src, indices_dst))

        ### add bond & feature information to graph
        G.ndata['x'] = torch.unsqueeze(x_sample, dim=1)  # [N, 1, 3]
        avg = x_0.mean(dim=0)
        G.ndata['x'] -= avg


        G.edata['d'] = G.ndata['x']

        G.ndata['v'] = G.ndata['x']


        return G, label_0

