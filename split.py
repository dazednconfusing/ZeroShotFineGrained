
import scipy.io as sio
import numpy as np
import pickle


easy_split = 'data/CUB2011/train_test_split_easy.mat'
hard_split = 'data/CUB2011/train_test_split_hard.mat'

easy_m = sio.loadmat(easy_split)
hard_m = sio.loadmat(hard_split)

train_cid, test_cid = easy_m['train_cid'], easy_m['test_cid']
print(train_cid.shape) # id of the classes
print(np.unique(train_cid))
print(test_cid.shape)
print(np.unique(test_cid))

test_classes_path = 'data/CUB_baseline/testclasses_id.mat'
test_att2label = np.squeeze(np.array(sio.loadmat(test_classes_path)['testclasses_id']))

print(test_att2label)

path = 'data/CUB1_data/original_att_splits.mat'
att_splits = sio.loadmat(path)
print(att_splits.keys())
print(att_splits['train_loc'].shape)
print(att_splits['trainval_loc'].shape)
print(len(np.unique(att_splits['train_loc'])))

# baseline paper uses easy dataset


