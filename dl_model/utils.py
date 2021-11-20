import os
import numpy as np
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve


root = ''
histones = ['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3']


def ROC(label, pred):
    if len(np.unique(np.array(label).reshape(-1))) == 1:
        print('all the labels are the same!')
        return 0
    else:
        label = np.array(label).reshape(-1)
        pred = np.array(pred).reshape(-1)
        return roc_auc_score(label, pred)


def PRC(label, pred):
    if len(np.unique(np.array(label).reshape(-1))) == 1:
        print('all the labels are the same!')
        return 0
    else:
        label = np.array(label).reshape(-1)
        pred = np.array(pred).reshape(-1)
        precision, recall, _ = precision_recall_curve(label, pred)
        return precision, recall, auc(recall, precision)


def metricsC(lab, pred, task, fn, loss=None):  # save the dict of ndarray
    if not os.path.exists(root):
        os.mkdir(root)
    base = os.path.join(root, 'CPRC')
    if not os.path.exists(base):
        os.mkdir(base)
    fn = task + '_' + fn
    save_path = os.path.join(base, fn)
    if task == 'VALID':
        color = '\033[0;34m'
    elif task == 'TEST':
        color = '\033[0;33m'
    roc_dict = {}
    prc_dict = {}
    precision_dict = {}
    recall_dict = {}
    save_dict = {}
    for i in range(len(histones)):
        roc_dict[histones[i]] = ROC(lab[:,i],pred[:,i])
        precision, recall, prc_dict[histones[i]] = PRC(lab[:,i],pred[:,i])
        precision_dict[histones[i]], recall_dict[histones[i]] = map(lambda x: x.tolist(), (precision, recall))
    if loss is not None:
        loss_str = ',Loss : %.4f' % loss
    else:
        loss_str = ''
    print('-'*25 + task + '-'*25)
    print((color + '%s\tTotalMean\tROC : %.4f\tPRC : %.4f\t%s\033[0m') % (task, np.mean(list(roc_dict.values())), np.mean(list(prc_dict.values())), loss_str))
    for histone in histones:
        print((color + '%s\t%s\tROC : %.4f\tPRC : %.4f\033[0m') % (task, histone.ljust(8), roc_dict[histone], prc_dict[histone]))
    save_dict['roc'] = roc_dict
    save_dict['prc'] = prc_dict
    np.save(save_path + '.npy', save_dict)
    return roc_dict, prc_dict


def metricsM(lab, pred, task, fn, loss=None):  # save ndarray
    if not os.path.exists(root):
        os.mkdir(root)
    base = os.path.join(root, 'MPRC')
    if not os.path.exists(base):
        os.mkdir(base)
    fn = task + '_' + fn
    save_path = os.path.join(base, fn)
    if task == 'VALID':
        color = '\033[0;36m'
    elif task == 'TEST':
        color = '\033[0;33m'
    save_dict = {}
    roc = ROC(lab, pred)
    precision, recall, prc = PRC(lab, pred)
    precision, recall = map(lambda x: x.tolist(), (precision, recall))
    if loss is not None:
        loss_str = ',Loss : %.4f' % loss
    else:
        loss_str = ''
    print('-'*25 + task + '-'*25)
    print((color + '%s\tROC : %.4f\tPRC : %.4f\t%s\033[0m') % (task, roc, prc, loss_str))
    save_dict['precision'] = precision
    save_dict['recall'] = recall
    np.save(save_path + '.npy', save_dict)
    return roc, prc

