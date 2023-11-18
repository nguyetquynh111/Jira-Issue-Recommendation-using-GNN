import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import torch
from metrics import CosineSimilarity
from retrieval_utils import *
DATA_DIR = 'data/FLUME'
geb_all = torch.load('all_geb.pt')


with open(f"{DATA_DIR}/train_scene_graphs.json", 'r') as file:
    train_scene_graphs = json.load(file)
with open(f"{DATA_DIR}/test_scene_graphs.json", 'r') as file:
    test_scene_graphs = json.load(file)

# Create labels
test_links = pd.read_csv(
    f"{DATA_DIR}/test_links.csv", keep_default_na=False)
test_issues = pd.read_csv(f"{DATA_DIR}/test_issues.csv", index_col="Unnamed: 0").index
match_test_links = test_links[test_links["relation"]!="None"]
y_test = []
filter_test_issues = []
for test_issue in tqdm(test_issues):
  filter_links = match_test_links[(match_test_links["key_1"]==test_issue)|(match_test_links["key_2"]==test_issue)]
  if len(filter_links)>0:
    match_issues = set(list(filter_links["key_1"].values) + list(filter_links["key_2"].values))
    match_issues.remove(test_issue)
    y_test.append(list(match_issues))
    filter_test_issues.append(test_issue)
test_issues = filter_test_issues

all_scene_graphs = train_scene_graphs
all_scene_graphs.update(test_scene_graphs)
all_scene_graphs = dict(sorted(all_scene_graphs.items()))
key2index = dict(zip(list(all_scene_graphs.keys()), range(len(all_scene_graphs))))
index2key = dict(zip(range(len(all_scene_graphs)), list(all_scene_graphs.keys())))

recommend_results = []
for key in tqdm(test_issues):
   geb = geb_all[[key2index.get(key)]*(len(all_scene_graphs)-1)]
   other_issues = list(range(len(all_scene_graphs)))
   other_issues.remove(key2index.get(key)) #remove current test issue
   other_issues_geb = geb_all[other_issues]
   scores = CosineSimilarity(torch.tensor(geb), torch.tensor(other_issues_geb))
   issues_key =  [index2key[i] for i in other_issues]
   compare_dict = list(zip(scores, issues_key))
   recommend_results.append([pair[1] for pair in sorted(compare_dict, reverse=True)])

for k in [1,2,3,5,10]:
  recommend_list = [i[:k] for i in recommend_results]
  acc, mrr, precision, recall = evaluate(recommend_list, y_test)
  print(f"Top {k}:")
  print(f"Acc = {acc}")
  print(f"MRR = {mrr}")
  print(f"Recall = {recall}")




