import numpy as np
import torch
import random
from model import *
from utils import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import LabelBinarizer


def collate(batch):
    dna, dnase, label, masked = [], [], [], []
    for item in batch:
        nan_sum = sum(map(lambda x: np.isnan(item[x]).sum(), ('dna', 'dnase', 'label', 'masked')))
        if nan_sum:
            continue
        dna.append(item['dna'])
        dnase.append(item['dnase'])
        label.append(item['label'])
        masked.append(item['masked'])
    dna, dnase, label, masked = map(lambda x: torch.from_numpy(np.array(x)), (dna, dnase, label, masked))
    return {
        'dna': dna,
        'dnase': dnase,
        'label': label,
        'masked': masked
    }


class HistoneDataset(Dataset):
    def __init__(self, path):
        self.dataset = np.load(path, allow_pickle=True)
        self.dna_set = self.dataset['dna']
        self.dnase_set = self.dataset['dnase']
        self.label_set = self.dataset['tlabel']
        self.peak_set = self.dataset['peaks']
        self.base_seed = LabelBinarizer()
        self.base_seed.fit(['A', 'C', 'G', 'T'])

    def __len__(self):
        return len(self.dna_set)

    def __getitem__(self, index):
        dna = self.dna_set[index]
        dna = dna.upper()
        dna_matrix = self.base_seed.transform(list(dna))
        dna_matrix = dna_matrix.transpose()
        dna_matrix = dna_matrix[np.newaxis, :]
        dnase = self.dnase_set[index]
        label = self.label_set[index]
        peak = self.peak_set[index]
        sample = {
            'dna': dna_matrix,
            'dnase': dnase,
            'label': label,
            'masked': self.maskDNA(peak)
        }
        return sample

    def maskDNA(self, peak):
        masked = np.zeros((1, 1000))
        for row in range(peak.shape[0]):
            masked[:, peak[row][0]:peak[row][1]] = 1
        return masked


class ModuleOperations():
    def __init__(self, drop_rate, learning_rate, batch_size, device, ):
        self.module = MyHistone(drop_rate, learning_rate, device)
        self.dataset = None  # current dataset
        self.idxs = None
        self.batch_size = batch_size  # batch size
        self.early_stop = 0  # memorize early stop time
        self.best_loss_c = np.float('Inf')
        self.best_prc_c = 0
        self.best_loss_m = np.float('Inf')
        self.best_prc_m = 0
        
    def prepare(self, data_path):
        self.dataset = HistoneDataset(data_path)
        self.idxs = list(range(len(self.dataset)))

    def shuffle(self):
        random.shuffle(self.idxs)  # shuffle index

    def train_module(self):
        sampler = SubsetRandomSampler(self.idxs)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0, drop_last=True, collate_fn=collate)
        for _, sample in enumerate(loader):
            self.module.train_module(sample['dna'], sample['dnase'], sample['label'], sample['masked'])

    def valid_module(self, fn):
        losses_c, labs_c, preds_c, losses_m, labs_m, preds_m = [], [], [], [], [], []  # initialize
        sampler = SubsetRandomSampler(self.idxs)  # rules of data sample and data load
        loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0, drop_last=True, collate_fn=collate)
        for _, sample in enumerate(loader):
            valid_rst = self.module.valid_module(sample['dna'], sample['dnase'], sample['label'], sample['masked'])
            if not valid_rst:
                continue
            loss_c, loss_m, pred_c, pred_m = valid_rst
            losses_c.append(loss_c)
            losses_m.append(loss_m)
            _n, _h, _w = sample['label'].size()
            lab = sample['label'].view(_n, _w)
            labs_c.extend(lab)
            preds_c.extend(pred_c)
            labs_m.extend(sample['masked'])
            preds_m.extend(pred_m)
        mean_c, prc_c = self.vision(labs_c, preds_c, 'classify', 'VALID', fn, losses_c)
        mean_m, prc_m = self.vision(labs_m, preds_m, 'match', 'VALID', fn, losses_m)
        return mean_c, prc_c, mean_m, prc_m

    def test_module(self, fn):
        labs_c, preds_c, labs_m, preds_m = [], [], [], []
        sampler = SubsetRandomSampler(self.idxs)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0, drop_last=True, collate_fn=collate)
        for _, sample in enumerate(loader):
            pred_c, pred_m = self.module.test_module(sample['dna'], sample['dnase'])
            _n, _h, _w = sample['label'].size()
            lab = sample['label'].view(_n, _w)
            labs_c.extend(lab)
            preds_c.extend(pred_c)
            labs_m.extend(sample['masked'])
            preds_m.extend(pred_m)
        mean_c, _ = self.vision(labs_c, preds_c, 'classify', 'TEST', fn)
        mean_m, _ = self.vision(labs_m, preds_m, 'match', 'TEST', fn)
        return mean_c, mean_m

    def update_module(self, save_path, mean_c, prc_c, mean_m, prc_m):
        if np.mean(list(prc_c.values())) > self.best_prc_c:
            self.best_prc_c = max(np.mean(list(prc_c.values())), self.best_prc_c)
            self.save_module(save_path)
        if prc_m > self.best_prc_m:
            self.best_prc_m = max(prc_m, self.best_prc_m)
            file_name, file_suffix = save_path.split('.')
            save_path_task2 = file_name + '-match.' + file_suffix;
            self.save_module(save_path_task2)
        if mean_c < self.best_loss_c or mean_m < self.best_loss_m:
            self.early_stop = 0 
            self.best_loss_c = min(mean_c, self.best_loss_c)
            self.best_loss_m = min(mean_m, self.best_loss_m)
        else:
            self.early_stop += 1
        if self.early_stop >= 5:
            return False
        return True

    def reset(self, best_loss_c, best_prc_c, best_loss_m, best_prc_m):
        self.early_stop = 0  # memorize early stop time
        self.best_loss_c = best_loss_c
        self.best_prc_c = best_prc_c
        self.best_loss_m = best_loss_m
        self.best_prc_m = best_prc_m

    def save_module(self, path):
        self.module.save_module(path)
    
    def load_module(self, path):
        self.module.load_module(path)

    def vision(self, labs, preds, task, vtype, fn, loss=None):
        mean = None
        if loss:
            loss = torch.tensor(loss)
            mean = torch.mean(loss)
        labs, preds = map(lambda x: np.array([item.numpy() for item in x]), (labs, preds))
        if task == 'classify':
            _, prc = metricsC(labs, preds, vtype, fn, mean)
        elif task == 'match':
            _, prc = metricsM(labs, preds, vtype, fn, mean)
        return mean, prc
