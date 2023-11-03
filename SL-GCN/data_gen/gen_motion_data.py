import os
import numpy as np
from numpy.lib.format import open_memmap

sets = {
    'train', 'val', 'test'

}

datasets = {
    'sign/27_2'
}

parts = {
    'joint', 'bone'
}
from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        for part in parts:
            print(dataset, set, part)
            data = np.load('/home/Student/s4582342/CVPR21Chal-SLR/SL-GCN/data/{}/{}_data_{}.npy'.format(dataset, set, part))
            N, C, T, V, M = data.shape
            print(data.shape)
            fp_sp = open_memmap(
                '/home/Student/s4582342/CVPR21Chal-SLR/SL-GCN/data/{}/{}_data_{}_motion.npy'.format(dataset, set, part),
                dtype='float32',
                mode='w+',
                shape=(N, C, T, V, M))
            for t in tqdm(range(T - 1)):
                fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
            fp_sp[:, :, T - 1, :, :] = 0
