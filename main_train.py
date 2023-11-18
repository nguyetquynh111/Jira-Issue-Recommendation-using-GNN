from Controller_GCN import Trainer, Evaluator
import torch
import os
import joblib
import json
import pandas as pd

DATA_DIR = 'data/FLUME'

info_dict = dict()
info_dict['save_dir'] = './Report'


info_dict['numb_sample'] = None # training sample for 1 epoch
info_dict['numb_epoch'] = 10 # number of epoch
info_dict['numb_gcn_layers'] = 1 # number of gcn layers to be stacked
info_dict['gcn_hidden_dim'] = [] # hidden layer in each gin layer
info_dict['gcn_output_dim'] = 1024 # graph embedding final dim
info_dict['gcn_input_dim'] = 2048 # node and edges dim of a graph
info_dict['batchnorm'] = True
info_dict['batch_size'] = 128
info_dict['dropout'] = 0.5
info_dict['visual_ft_dim'] = 2048
info_dict['optimizer'] = 'Adam' # or Adam
info_dict['learning_rate'] = 3e-4
info_dict['activate_fn'] = 'swish' # swish, relu, leakyrelu
info_dict['grad_clip'] = 2 # Gradient clipping
# info_dict['use_residual'] = False # always set it to false (not implemented yet)
# Embedder for each objects and predicates, embed graph only base on objects
info_dict['model_name'] = 'GCN_ObjAndPredShare_NoFtExModule_LSTM' 
info_dict['checkpoint'] = None # Training from a pretrained path
info_dict['margin_matrix_loss'] = 0.35
info_dict['rnn_numb_layers'] = 2
info_dict['rnn_bidirectional'] = True
info_dict['rnn_structure'] = 'LSTM' # LSTM or GRU (LSTM gave better result)
info_dict['graph_emb_dim'] = info_dict['gcn_output_dim']*2
info_dict['include_pred_ft'] = True # include visual predicate features or not
info_dict['freeze'] = False # Freeze all layers except the graph convolutional network and graph embedding module

with open(f"{DATA_DIR}/train_scene_graphs.json", 'r') as file:
    val_scene_graphs = json.load(file)
val_links = pd.read_csv(f"{DATA_DIR}/val_links.csv")

def run_train(info_dict):
    if not os.path.exists(info_dict['save_dir']):
        print(f"Creating {info_dict['save_dir']} folder")
        os.makedirs(info_dict['save_dir'])
        
    trainer = Trainer(info_dict)
    trainer.train()
    
    scene_graphs, issues_1, issues_2 = trainer.get_val_data()
    lossVal = trainer.validate_retrieval(scene_graphs, issues_1, issues_2)
    info_txt = f"Loss Val last training: {lossVal}"   
    print(info_txt)

    info_dict['checkpoint'] = 'all_pretrained_model.pt'
    evaluator = Evaluator(info_dict)
    evaluator.load_trained_model()

    lossVal = evaluator.validate_retrieval(scene_graphs, issues_1, issues_2)
    info_txt = f"Loss Val last evaluate: {lossVal}"   
    print(info_txt)

run_train(info_dict)