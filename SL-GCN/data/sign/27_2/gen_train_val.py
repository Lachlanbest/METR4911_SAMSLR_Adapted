import numpy as np

parts = {'joint', 'bone', 'joint_motion', 'bone_motion'}

for part in parts:
    print(part)
    data_train = np.load('/home/Student/s4582342/CVPR21Chal-SLR/SL-GCN/data/sign/27_2/train_data_{}.npy'.format(part))
    data_val = np.load('/home/Student/s4582342/CVPR21Chal-SLR/SL-GCN/data/sign/27_2/val_data_{}.npy'.format(part))

    data_train_val = np.concatenate((data_train, data_val), axis=0)
    print(data_train_val.shape)


    np.save('/home/Student/s4582342/CVPR21Chal-SLR/SL-GCN/data/sign/27_2/train_val_data_{}.npy'.format(part), data_train_val)