'''
Calculate recall@k
'''
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.ranking import roc_auc_score
import torch.nn.functional as F
device = torch.device('cuda:0')


def l2norm(X, ds_1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=ds_1, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def calculate_metric(gtnp, pdnp):
    # input are numpy vector
    o_pdnp = np.copy(pdnp)  # this is for AUROC score
    pdnp[pdnp >= 0.5] = 1
    pdnp[pdnp != 1] = 0
    total_samples = len(gtnp)
    # print(f"Total sample: {total_samples}")
    total_correct = np.sum(gtnp == pdnp)
    accuracy = total_correct / total_samples
    gt_pos = np.where(gtnp == 1)[0]
    gt_neg = np.where(gtnp == 0)[0]
    TP = np.sum(pdnp[gt_pos])
    TN = np.sum(1 - pdnp[gt_neg])
    FP = np.sum(pdnp[gt_neg])
    FN = np.sum(1 - pdnp[gt_pos])
    precision = TP / (TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    metrics = {}
    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['tp'] = int(TP)
    metrics['tn'] = int(TN)
    metrics['fp'] = int(FP)
    metrics['fn'] = int(FN)
    try:
        metrics['auc'] = roc_auc_score(gtnp, o_pdnp)
    except Exception as e:
        print(e)
        metrics['auc'] = -1
    return metrics


def find_recall(list_query, list_source, result_matching, label_matching, k=1):
    '''
    list_query, list_source: id of query and source (list)
    result_matching: index of above list predict to be matching to each other (list of list)
    label_matching: groundtruth id matching between query: source (dict)
    '''
    assert len(list_query) == len(result_matching)
    result = []
    for idx, query in enumerate(list_query):
        flag = 0
        topk = result_matching[idx][:k]
        query_idx = topk[0][0]
        source_idx = [x[1] for x in topk]
        assert query_idx == idx
        query_id = list_query[query_idx]
        source_id = [list_source[x] for x in source_idx]
        label_id = label_matching[query_id]
        for sid in source_id:
            if sid in label_id:
                flag = 1
                break
        result.append(flag)
    result = np.asarray(result)
    recall = np.sum(result) / len(result)
    return recall

def CosineSimilarity(issues_1_geb, issues_2_geb):
    dot_product = torch.matmul(issues_1_geb, issues_2_geb.t())
    norm_A = torch.norm(issues_1_geb, dim=1, keepdim=True)
    norm_B = torch.norm(issues_2_geb, dim=1, keepdim=True)

    similarities = dot_product / (norm_A * norm_B.t())
    similarities = similarities.diagonal()
    return similarities


class ContrastiveLoss_CosineSimilarity(nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss_CosineSimilarity, self).__init__()
        self.max_violation = max_violation
        self.margin = margin

    def forward(self, issues_1_geb, issues_2_geb, labels):
        scores = CosineSimilarity(issues_1_geb, issues_2_geb)
        labels = torch.tensor(labels, dtype=torch.float32)
        return F.mse_loss(scores.to(device), labels.to(device))