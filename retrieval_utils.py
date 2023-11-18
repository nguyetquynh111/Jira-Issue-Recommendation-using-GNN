import numpy as np
import torch
from metrics import CosineSimilarity
from collections import OrderedDict
from torch.autograd import Variable


def evalrank(issue_1_geb, issue_2_geb):
    print('Contrastive Score ...')
    cos_sim = CosineSimilarity(torch.tensor(issue_1_geb), torch.tensor(issue_2_geb))
    return cos_sim.mean()

def Acc(pred, gt):
	acc = 0
	for i, item in enumerate(pred):
		if item in gt:
			acc += 1.0
			break
	return acc

def MRR(pred, gt):
	mrr = 0
	for i, item in enumerate(pred):
		if item in gt:
			mrr += 1.0/(i+1)
	return mrr

def Precision_Recall(pred, gt):
	right = 0

	for item in gt:
		if item in pred: # relevant
			right+=1

	if len(pred) == 0:
		precision = 0
	else:
		precision = right/len(pred)
	recall = right/len(gt)

	return precision, recall

def metrics(recommend, label):
  acc = 0
  mrr = 0
  precision = 0
  recall = 0
  for i in range(0, len(recommend)):
    if len(label[i])!=0:
      acc+=Acc(recommend[i], label[i])
      mrr+=MRR(recommend[i], label[i])
      precision_recall = Precision_Recall(recommend[i], label[i])
      precision+=precision_recall[0]
      recall+=precision_recall[1]
  return acc/(len(recommend)), mrr/(len(recommend)), precision/(len(recommend)), recall/(len(recommend))

def evaluate(recommend, labels):
    acc = 0
    mrr = 0
    precision = 0
    recall = 0
    for i in range(0, len(recommend)):
        acc += Acc(recommend[i], labels[i])
        mrr += MRR(recommend[i], labels[i])
        precision_recall = Precision_Recall(recommend[i], labels[i])
        precision += precision_recall[0]
        recall += precision_recall[1]
    return acc/(len(recommend)), mrr/(len(recommend)), precision/(len(recommend)), recall/(len(recommend))
