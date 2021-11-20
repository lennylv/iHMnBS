import torch
import random
import os
from train import *
import numpy as np


EPOCH = 20
DROP_RATE = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 256


def get_fn(path, fl):    
    for fn in os.listdir(path):
        cur_path = os.path.join(path, fn)
        if os.path.isdir(cur_path):
            get_fn(cur_path, fl)
        else:
            fl.append(cur_path)

def main():
    data_paths = []
    get_fn('', data_paths)
    data_paths = sorted(data_paths, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    if torch.cuda.is_available():
        device_ids = [3, 2, 1, 0]
    else:
        device_ids = None
    base = ''
    if not os.path.exists(base):
        os.mkdir(base)
    test_set = data_paths[:1]
    print(test_set)
    train_valid_set = data_paths[1:]
    print(train_valid_set)
    tissue = data_paths[0].split('.')[0].split('/')[-1].split('_')[0]
    save_path = os.path.join(base, 'model-'+tissue+'.txt')  # save path of current best model
    curve_lc, curve_lm = [], []  # coordinates of loss(c) and loss(m)
    torch.multiprocessing.set_sharing_strategy('file_system')
    model = ModuleOperations(DROP_RATE, LEARNING_RATE, BATCH_SIZE, device_ids, )
    for i in range(30, 50):
        random.shuffle(train_valid_set)
        valid_set = train_valid_set[:2]
        train_set = train_valid_set[2:]
        for data_path in train_set:
            model.prepare(data_path)
            model.shuffle()
            model.train_module()
        for idx, data_path in enumerate(valid_set):
            model.prepare(data_path)
            model.shuffle()
            mean_c, prc_c, mean_m, prc_m = model.valid_module(tissue+'_epoch'+str(i)+'_'+str(idx))
        model.load_module(save_path)  # load model
        print('Epoch {} is over. Current Tissue Is {}'.format(i, tissue))
        if not mark:  # early stop time greater than 5
            break
    for data_path in test_set:
        print(data_path)
        model.load_module(save_path)
        model.prepare(data_path)
        model.test_module(tissue)
    model.save_module(os.path.join(base, 'model-finish.txt'))


if __name__ == '__main__':
    main()