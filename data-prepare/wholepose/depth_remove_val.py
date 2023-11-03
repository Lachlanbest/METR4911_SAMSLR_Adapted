import os

input_path = '/home/Student/s4582342/val'

for root, _, fnames in os.walk(input_path):
    for fname in fnames:
        if '.npy' in fname:
            file_path = os.path.join(root, fname)
            os.remove(file_path)