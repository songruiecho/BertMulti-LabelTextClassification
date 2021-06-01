import pickle
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import config
from pytorch_pretrained_bert import BertModel,BertTokenizer

tokenizer = BertTokenizer.from_pretrained(config.bert_path+'/vocab.txt')

def trans_label2n_hot(labels, hot_length):
    ''' use to trans multi-label to n-hot vectors
    :param labels:
    :return:
    '''
    zeors = np.array([0]*hot_length)
    zeors[labels] = 1
    return zeors

def cal_acc(pred, label):
    correct = pred - label
    # counts = label.shape[0]*label.shape[1]
    counts = sum(sum(label != 0))
    wrong = sum(sum(correct==-1))
    acc = 1.0*(counts-wrong)/counts
    return acc

def load_data():
    with open('data/data.txt', 'r', encoding='utf-8') as rf:
        datas = [each.strip().split('\t') for each in rf.readlines()]
    # process data to Bert input
    Datas = []
    for data in datas:
        labels = sorted([int(a) for a in data[0].split('-')])
        labels = trans_label2n_hot(labels, 35)
        sentence = '[CLS]' + data[1].replace(' ','').replace('<SEP>','[SEP]')
        tokens = tokenizer.tokenize(sentence)[:config.max_len]
        if len(tokens) < config.max_len:        # padding for max length
            tokens.extend(['[PAD]']*(config.max_len-len(tokens)))
        ids = np.array(tokenizer.convert_tokens_to_ids(tokens))
        labels = torch.from_numpy(labels)
        ids = torch.from_numpy(ids)
        Datas.append([ids, labels])
    split = int(len(Datas)*config.train_test_split)
    Trains = Datas[:split]
    Tests = Datas[split:]   # split+193
    train_loader = DataLoader(Trains, config.batch_size, shuffle=True)
    test_loader = DataLoader(Tests, config.batch_size, shuffle=True)
    print('data load finished! {}\t{}'.format(len(Trains), len(Tests)))
    return train_loader, test_loader

