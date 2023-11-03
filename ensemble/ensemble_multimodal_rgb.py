import argparse
import pickle

import numpy as np
from tqdm import tqdm



# Load label data
label = open('/home/Student/s4582342/CVPR21Chal-SLR/ensemble/gcn/test_labels_pseudo.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('/home/Student/s4582342/CVPR21Chal-SLR/ensemble/gcn/test_gcn_w_val_finetune.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r4 = open('/home/Student/s4582342/CVPR21Chal-SLR/ensemble/test_feature_w_val_finetune.pkl', 'rb')
r4 = list(pickle.load(r4).items())

alpha = [1, 0.4]  # Weights for r1 and r4

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0
with open('/home/Student/s4582342/CVPR21Chal-SLR/ensemble/predictions_rgb.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name4, r44 = r4[i]
        assert name == name1 == name4
        mean += r11.mean()
        score = (r11 * alpha[0] + r44 * alpha[1]) / np.array(alpha).sum()
        score = score.squeeze()
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(score)
        scores.append(score)
        preds.append(r)
        right_num += int(r == int(l))
        total_num += 1
        f.write('{}, {}\n'.format(name, r))
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print('top1: ', acc)
    print('top5: ', acc5)

f.close()
print(mean / len(label[0]))

# with open('./val_score.pkl', 'wb') as f:
#     score_dict = dict(zip(names, scores))
#     pickle.dump(score_dict, f)