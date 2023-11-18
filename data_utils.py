# INCLUDE VISUAL FEATURES FOR PREDICATE (optional)
from PIL import Image
from functools import partial
import joblib
import random
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json
import pandas as pd
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def indexing_sent(sent, word2index, add_start_end=True):
    words = word_tokenize(sent)
    if add_start_end:
        words = ['<start>'] + words + ['<end>']
    words_idx = []
    for word in words:
        try:
            idx = word2index[word]
        except:
            idx = word2index['<unk>']
        words_idx.append(idx)
    return words_idx


def indexing_rels(rels, word2index, add_start_end=True):
    rels_idx = []
    for rel in rels:
        for idx, word in enumerate(rel):
            if ':' in word:  # rels in sentence has ":"
                word = word.split(':')[0]
                rel[idx] = word
        rel = ' '.join(rel)
        rel = word_tokenize(rel)
        if add_start_end:
            rel = ['<start>'] + rel + ['<end>']
        rel_idx = []
        for word in rel:
            try:
                idx = word2index[word]
            except:
                idx = word2index['<unk>']
            rel_idx.append(idx)
        rels_idx.append(rel_idx)
    return rels_idx


def encode_scene_graphs_to_matrix(scene_graph, word2index):
    '''
    scene_graph is dict with text features and relations (rels)
    word2index dictionary to encode word into numeric
    Return obj, pred, and edge matrix in which
    obj = [n_obj, ] indicating index of object in obj_to_idx --> will pass to embedding
    pred = [n_pred, ] indicating index of predicate in pred_to_idx --> will pass to embedding
    edge = [n_pred, 2] indicating the relations between objects where edge[k] = [i,j] = [obj[i], pred[k], obj[j]] relations
    sent_to_idx: encoded sentence with <start> and <end> token
    text: original text features
    '''

    obj_np = []
    pred_np = []
    edge_np = []

    sent_to_idx = indexing_sent(
        sent=scene_graph['text'], word2index=word2index, add_start_end=True)  # list

    labels = [x[0] for x in scene_graph['rels']] + [x[2]
                                                    for x in scene_graph['rels']]
    labels = np.unique(np.asarray(labels)).tolist()

    for idx, obj in enumerate(labels):
        try:
            label_to_idx = word2index[obj]
        except:
            label_to_idx = word2index['<unk>']
        obj_np.append(label_to_idx)

    for idx, rel in enumerate(scene_graph['rels']):
        sub, pred_label, obj = rel[0], rel[1], rel[2]
        sub_pos = labels.index(sub)
        obj_pos = labels.index(obj)
        edge_np.append([int(sub_pos), int(obj_pos)])

    pred_np = indexing_rels(
        rels=scene_graph['rels'], word2index=word2index, add_start_end=True)  # list of list
    # pred: [<start> , sub , pred, obj, <end>]
    # len of a pred <start> sub, pred (can be multiple words), obj <end>
    len_pred = [len(x) for x in pred_np]
    obj_np = np.asarray(obj_np, dtype=int)
    edge_np = np.asarray(edge_np, dtype=int)

    # obj and edge is numpy array, other is lis
    return obj_np, pred_np, edge_np, len_pred, sent_to_idx


# ====== Issue DATASET ======
# Only use for validating entire dataset
# Generate issue scene_graphs dataset only (sentence + scene_graphs)
class IssueDataset(Dataset):
    def __init__(self, scene_graphs, word2index, numb_sample=None):
        # Do something
        self.scene_graphs = scene_graphs
        self.list_issue_key = list(self.scene_graphs.keys())
        self.numb_sample = numb_sample
        self.word2index = word2index
        if self.numb_sample is None or self.numb_sample <= 0 or self.numb_sample > len(self.scene_graphs):
            self.numb_sample = len(self.scene_graphs)
            assert self.numb_sample == len(self.list_issue_key)

    def __len__(self):
        return self.numb_sample

    def __getitem__(self, idx):
        issue_key = self.list_issue_key[idx]
        issue_obj_np, issue_pred_np, issue_edge_np, issue_len_pred, issue_sent_np = encode_scene_graphs_to_matrix(
            scene_graph=self.scene_graphs[issue_key], word2index=self.word2index)

        result = dict()
        result['object'] = issue_obj_np
        result['predicate'] = issue_pred_np
        result['edge'] = issue_edge_np
        result['sent'] = issue_sent_np
        result['numb_obj'] = len(issue_obj_np)
        result['numb_pred'] = len(issue_pred_np)
        result['len_pred'] = issue_len_pred

        return result


def issue_collate_fn(batch):
    issue_obj = np.array([])
    issue_pred = []
    issue_edge = np.array([])
    issue_numb_obj = []
    issue_numb_pred = []
    issue_sent = []
    issue_len_sent = []
    issue_len_pred = []

    for ba in batch:
        issue_obj = np.append(issue_obj, ba['object'])
        for idx_row in range(ba['edge'].shape[0]):
            edge = ba['edge'][idx_row]
            issue_edge = np.append(issue_edge, edge)
            issue_pred += [torch.LongTensor(ba['predicate'][idx_row])]

        issue_numb_obj += [ba['numb_obj']]
        issue_numb_pred += [ba['numb_pred']]
        issue_sent += [torch.LongTensor(ba['sent'])]
        issue_len_sent += [len(ba['sent'])]
        issue_len_pred += ba['len_pred']
    issue_edge = issue_edge.reshape(-1, 2)

    issue_obj = torch.LongTensor(issue_obj)
    issue_edge = torch.LongTensor(issue_edge)
    issue_numb_obj = torch.LongTensor(issue_numb_obj)
    issue_numb_pred = torch.LongTensor(issue_numb_pred)

    return issue_obj, issue_pred, issue_edge,  issue_sent, \
        issue_numb_obj, issue_numb_pred, issue_len_pred, issue_len_sent


def make_IssueDataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=issue_collate_fn,
                            pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader


# ====== ISSUE-ISSUE TRIPLET DATASET ======
class PairGraphPrecomputeDataset(Dataset):
    '''
    Generate pair of graphs which from two issues
    '''

    def __init__(self, scene_graphs, word2index, train_links, numb_sample):
        '''
        scene_graph: dictionary of scene graph from images with format issue_1_scene_graphs[issue_1_id]['rels'] and issue_1_scene_graphs[issue_1_id]['labels']
        word2index: dictionary to map words into index for learning embedding
        numb_sample: int indicating number of sample in the dataset
        '''
        # Do something
        self.scene_graphs = scene_graphs
        self.train_links = train_links
        self.numb_sample = numb_sample
        self.word2index = word2index
        self.list_match_pairs = []
        number_linked_data = len(train_links[train_links['relation']!='None'])
        if self.numb_sample is None or self.numb_sample<=number_linked_data:
            self.numb_sample = number_linked_data*4

    # Have to run this function at the beginning of every epoch
    def create_pairs(self, seed=1509):
        # Shuffle Item
        random.seed(seed)
        print('Creating Pairs of Graphs ...')
        linked_data = self.train_links[self.train_links['relation']!='None']
        not_linked_data = self.train_links[self.train_links['relation']=='None']
        data = pd.concat([linked_data, not_linked_data.sample(self.numb_sample-len(linked_data))])
        data = data.sample(len(data))
        self.list_match_pairs = []
        for _, row in data.iterrows():
            issue_1, issue_2, relation = row
            self.list_match_pairs.append(
                (issue_1, issue_2, int(relation != "None")))
        self.numb_pairs = len(self.list_match_pairs)
        self.samples = self.list_match_pairs

    def __getitem__(self, i):
        # Get item
        sample = self.samples[i]
        issue_1, issue_2, relation = sample

        issue_obj_np_1, issue_pred_np_1, issue_edge_np_1, issue_len_pred_1, issue_sent_np_1 = encode_scene_graphs_to_matrix(
            scene_graph=self.scene_graphs[issue_1], word2index=self.word2index)
        issue_obj_np_2, issue_pred_np_2, issue_edge_np_2, issue_len_pred_2, issue_sent_np_2 = encode_scene_graphs_to_matrix(
            scene_graph=self.scene_graphs[issue_2], word2index=self.word2index)

        result = dict()
        result['issue_1'] = dict()
        result['issue_2'] = dict()

        # All is numpy array
        result['issue_1']['object'] = issue_obj_np_1
        result['issue_1']['predicate'] = issue_pred_np_1
        result['issue_1']['edge'] = issue_edge_np_1
        result['issue_1']['sent'] = issue_sent_np_1
        result['issue_1']['numb_obj'] = len(issue_obj_np_1)
        result['issue_1']['numb_pred'] = len(issue_pred_np_1)
        result['issue_1']['id'] = issue_1
        result['issue_1']['len_pred'] = issue_len_pred_1
        # All is list
        result['issue_2']['object'] = issue_obj_np_2
        result['issue_2']['predicate'] = issue_pred_np_2
        result['issue_2']['edge'] = issue_edge_np_2
        result['issue_2']['sent'] = issue_sent_np_2
        result['issue_2']['numb_obj'] = len(issue_obj_np_2)
        result['issue_2']['numb_pred'] = len(issue_pred_np_2)
        result['issue_2']['id'] = issue_2
        result['issue_2']['len_pred'] = issue_len_pred_2

        result['label'] = int(relation != "None")

        return result

    def __len__(self):
        return (len(self.samples))

# Collate function for preprocessing batch in dataloader


def pair_precompute_collate_fn(batch):
    '''
    image obj, pred, edge is tensor
    others is list
    '''
    issue_1_obj = np.array([])
    issue_1_pred = []
    issue_1_edge = np.array([])
    issue_1_numb_obj = []
    issue_1_numb_pred = []
    issue_1_len_pred = []
    issue_1_sent = []
    issue_1_len_sent = []

    issue_2_obj = np.array([])
    issue_2_pred = []
    issue_2_edge = np.array([])
    issue_2_numb_obj = []
    issue_2_numb_pred = []
    issue_2_len_pred = []
    issue_2_sent = []
    issue_2_len_sent = []

    issue_1_id = []  # for debug
    issue_2_id = []  # for debug
    labels = []

    for ba in batch:
        labels += [ba["label"]]
        # Issue 1 scene_graphs
        issue_1_obj = np.append(issue_1_obj, ba['issue_1']['object'])
        for idx_row in range(ba['issue_1']['edge'].shape[0]):
            edge = ba['issue_1']['edge'][idx_row]
            issue_1_pred += [torch.LongTensor(ba['issue_1']
                                              ['predicate'][idx_row])]
            issue_1_edge = np.append(issue_1_edge, edge)
        issue_1_numb_obj += [ba['issue_1']['numb_obj']]
        issue_1_numb_pred += [ba['issue_1']['numb_pred']]
        issue_1_sent += [torch.LongTensor(ba['issue_1']['sent'])]
        issue_1_len_sent += [len(ba['issue_1']['sent'])]
        issue_1_len_pred += ba['issue_1']['len_pred']

        # Issue 2 scene_graphs
        issue_2_obj = np.append(issue_2_obj, ba['issue_2']['object'])
        for idx_row in range(ba['issue_2']['edge'].shape[0]):
            edge = ba['issue_2']['edge'][idx_row]
            issue_2_pred += [torch.LongTensor(ba['issue_2']
                                              ['predicate'][idx_row])]
            issue_2_edge = np.append(issue_2_edge, edge)
        issue_2_numb_obj += [ba['issue_2']['numb_obj']]
        issue_2_numb_pred += [ba['issue_2']['numb_pred']]
        issue_2_sent += [torch.LongTensor(ba['issue_2']['sent'])]
        issue_2_len_sent += [len(ba['issue_2']['sent'])]
        issue_2_len_pred += ba['issue_2']['len_pred']

        issue_1_id += [ba['issue_1']['id']]
        issue_2_id += [ba['issue_2']['id']]

    # reshape edge to [n_pred, 2] size
    issue_1_edge = issue_1_edge.reshape(-1, 2)
    issue_2_edge = issue_2_edge.reshape(-1, 2)

    issue_1_obj = torch.LongTensor(issue_1_obj)
    issue_1_edge = torch.LongTensor(issue_1_edge)
    assert issue_1_edge.shape[0] == sum(issue_1_numb_pred)

    issue_2_obj = torch.LongTensor(issue_2_obj)
    issue_2_edge = torch.LongTensor(issue_2_edge)
    assert issue_2_edge.shape[0] == sum(issue_2_numb_pred)

    return issue_1_obj, issue_1_pred, issue_1_edge, issue_1_sent, issue_1_numb_obj, issue_1_numb_pred, issue_1_len_pred, issue_1_len_sent, \
        issue_2_obj, issue_2_pred, issue_2_edge, issue_2_sent, issue_2_numb_obj, issue_2_numb_pred, issue_2_len_pred, issue_2_len_sent, labels


def make_PairGraphPrecomputeDataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pair_precompute_collate_fn,
                            pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader
