# Precompute EfficientNet before, not run EfficientNet here
# Include the Predicate visual Ft
# Add Extra GCN for textual graph (after the RNN)
from data_utils import *
import models as md
from metrics import *
from retrieval_utils import *
from torch.nn.utils.rnn import pad_sequence
import itertools
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import time
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os
import torchvision.transforms as transforms
import json
import pandas as pd
non_blocking = True
device = torch.cuda.set_device(0)

DATA_DIR = 'data/FLUME'
MAXLEN = 256
with open(f"{DATA_DIR}/word2index.json", 'r') as file:
    word2index = json.load(file)
TOTAL_WORDS = len(word2index)

# Load train data
with open(f"{DATA_DIR}/train_scene_graphs.json", 'r') as file:
    train_scene_graphs = json.load(file)
train_links = pd.read_csv(f"{DATA_DIR}/train_links.csv", keep_default_na=False)

# Load val data
with open(f"{DATA_DIR}/test_scene_graphs.json", 'r') as file:
    test_scene_graphs = json.load(file)
val_links = pd.read_csv(f"{DATA_DIR}/val_links.csv", keep_default_na=False)


def padding_sequence(x):
    new_x = []
    for sent in x:
        if len(sent) > MAXLEN:
            new_x.append(sent[:MAXLEN])
        else:
            new_x.append(sent)
    return pad_sequence(new_x, batch_first=True)


def print_dict(di):
    result = ''
    for key, val in di.items():
        key_upper = key.upper()
        result += f"{key_upper}: {val}\n"
    return result


class Trainer():
    def __init__(self, info_dict):
        super(Trainer, self).__init__()
        ##### INIT #####
        self.info_dict = info_dict
        # 50000 - number of training sample in 1 epoch
        self.numb_sample = info_dict['numb_sample']
        self.numb_epoch = info_dict['numb_epoch']  # 10 - number of epoch
        self.unit_dim = 300
        # number of gin layer in graph embedding
        self.numb_gcn_layers = info_dict['numb_gcn_layers']
        # hidden layer in each gin layer
        self.gcn_hidden_dim = info_dict['gcn_hidden_dim']
        self.gcn_output_dim = info_dict['gcn_output_dim']
        self.gcn_input_dim = info_dict['gcn_input_dim']
        self.activate_fn = info_dict['activate_fn']
        self.grad_clip = info_dict['grad_clip']
        self.use_residual = False  # info_dict['use_residual']
        self.batchnorm = info_dict['batchnorm']
        self.dropout = info_dict['dropout']
        self.batch_size = info_dict['batch_size']
        self.save_dir = info_dict['save_dir']
        self.optimizer_choice = info_dict['optimizer']
        self.learning_rate = info_dict['learning_rate']
        self.model_name = info_dict['model_name']
        self.checkpoint = info_dict['checkpoint']
        self.margin_matrix_loss = info_dict['margin_matrix_loss']
        self.rnn_numb_layers = info_dict['rnn_numb_layers']
        self.rnn_bidirectional = info_dict['rnn_bidirectional']
        self.rnn_structure = info_dict['rnn_structure']
        self.visual_ft_dim = info_dict['visual_ft_dim']
        self.ge_dim = info_dict['graph_emb_dim']
        self.include_pred_ft = info_dict['include_pred_ft']
        self.freeze = info_dict['freeze']
        self.datatrain = PairGraphPrecomputeDataset(scene_graphs=train_scene_graphs,
                                                    word2index=word2index,
                                                    train_links=train_links,
                                                    numb_sample=self.numb_sample)

        # DECLARE MODEL
        self.gcn_model_cap = md.GCN_Network(gcn_input_dim=self.gcn_output_dim, gcn_pred_dim=self.gcn_output_dim,
                                            gcn_output_dim=self.gcn_output_dim, gcn_hidden_dim=self.gcn_hidden_dim,
                                            numb_gcn_layers=self.numb_gcn_layers, batchnorm=self.batchnorm,
                                            dropout=self.dropout, activate_fn=self.activate_fn, use_residual=False)

        self.embed_model_cap = md.WordEmbedding(numb_words=TOTAL_WORDS, embed_dim=self.unit_dim,
                                                sparse=False)

        self.sent_model = md.SentenceModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim,
                                           numb_layers=self.rnn_numb_layers,
                                           dropout=self.dropout, bidirectional=self.rnn_bidirectional,
                                           structure=self.rnn_structure)

        self.rels_model = md.RelsModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim,
                                       numb_layers=self.rnn_numb_layers,
                                       dropout=self.dropout, bidirectional=self.rnn_bidirectional,
                                       structure=self.rnn_structure)

        self.graph_embed_model = md.GraphEmb(node_dim=self.gcn_output_dim, edge_dim=self.gcn_output_dim,
                                             fusion_dim=self.ge_dim, activate_fn=self.activate_fn,
                                             batchnorm=self.batchnorm, dropout=self.dropout)

        self.embed_model_cap = self.embed_model_cap.to(device)
        self.sent_model = self.sent_model.to(device)
        self.rels_model = self.rels_model.to(device)
        self.gcn_model_cap = self.gcn_model_cap.to(device)
        self.graph_embed_model = self.graph_embed_model.to(device)

        if self.freeze:  # freeze most of component
            for p in self.embed_model_cap.parameters():
                p.requires_grad = False
            for p in self.sent_model.parameters():
                p.requires_grad = False
            for p in self.rels_model.parameters():
                p.requires_grad = False

        # PARAMS & OPTIMIZER
        self.params = []
        self.params += list(filter(lambda p: p.requires_grad,
                            self.embed_model_cap.parameters()))
        self.params += list(filter(lambda p: p.requires_grad,
                            self.sent_model.parameters()))
        self.params += list(filter(lambda p: p.requires_grad,
                            self.rels_model.parameters()))
        self.params += list(filter(lambda p: p.requires_grad,
                            self.gcn_model_cap.parameters()))
        self.params += list(filter(lambda p: p.requires_grad,
                            self.graph_embed_model.parameters()))

        if self.optimizer_choice.lower() == 'adam':
            self.optimizer = optim.Adam(self.params,
                                        lr=self.learning_rate,
                                        betas=(0.9, 0.999),
                                        eps=1e-08,
                                        weight_decay=0)

        if self.optimizer_choice.lower() == 'sgd':
            self.optimizer = optim.SGD(self.params,
                                       lr=self.learning_rate,
                                       momentum=0.9,
                                       weight_decay=0)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 30 epochs"""
        lr = self.learning_rate * \
            (0.1 ** (epoch // 15))  # 15 epoch update once
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # ---------- WRITE INFO TO TXT FILE ---------

    def extract_info(self):
        try:
            timestampLaunch = self.timestampLaunch
        except:
            timestampLaunch = 'undefined'
        model_info_log = open(
            f"{self.save_dir}/{self.model_name}-{timestampLaunch}-INFO.log", "w")
        result = f"===== {self.model_name} =====\n"
        result += print_dict(self.info_dict)
        model_info_log.write(result)
        model_info_log.close()
    
    def get_val_data(self, numb_sample=None):
        all_scene_graphs = train_scene_graphs
        all_scene_graphs.update(test_scene_graphs)
        all_scene_graphs = dict(sorted(all_scene_graphs.items()))
        key2index = dict(zip(list(all_scene_graphs.keys()), range(len(all_scene_graphs))))

        linked_data = val_links[val_links['relation']!="None"]
        if numb_sample==None or numb_sample<=len(linked_data):
            numb_sample = len(linked_data)*4
        not_linked_data = val_links[val_links['relation']=="None"].sample(numb_sample-len(linked_data))
        data = pd.concat([linked_data, not_linked_data])
        data = data.sample(len(data))

        issues_1 = []
        issues_2 = []
        for _, row in data.iterrows():
            key_1, key_2, _ = row
            issues_1.append(key2index.get(key_1))
            issues_2.append(key2index.get(key_2))
        return all_scene_graphs, issues_1, issues_2


    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        # ---- Load checkpoint
        if self.checkpoint is not None and not os.path.exists(self.checkpoint):
            self.checkpoint = None
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.embed_model_cap.load_state_dict(
                modelCheckpoint['embed_model_issue_state_dict'])
            self.sent_model.load_state_dict(
                modelCheckpoint['sent_model_state_dict'])
            self.rels_model.load_state_dict(
                modelCheckpoint['rels_model_state_dict'])
            self.gcn_model_cap.load_state_dict(
                modelCheckpoint['gcn_model_issue_state_dict'])
            self.graph_embed_model.load_state_dict(
                modelCheckpoint['graph_embed_model_state_dict'])
            if not self.freeze:
                self.optimizer.load_state_dict(
                    modelCheckpoint['optimizer_state_dict'])
        else:
            print("TRAIN FROM SCRATCH")

    # ---------- RUN TRAIN ---------

    def train(self):
        ## LOAD PRETRAINED MODEL ##
        self.load_trained_model()

        scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5,
                                      mode='min', verbose=True, min_lr=1e-6)
        # scheduler_remaining_models = ReduceLROnPlateau(self.optimizer_remaining_models, factor = 0.5, patience=10,
        # mode = 'min', verbose=True, min_lr=1e-6)

        ## LOSS FUNCTION ##
        loss_geb = ContrastiveLoss_CosineSimilarity(
            margin=self.margin_matrix_loss, max_violation=True)
        loss_geb = loss_geb.to(device)

        ## REPORT ##
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        self.timestampLaunch = timestampDate + '-' + timestampTime
        # f_log = open(f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}-REPORT.log", "w")
        writer = SummaryWriter(
            f'{self.save_dir}/{self.model_name}-{self.timestampLaunch}/')
        self.extract_info()

        ## TRAIN THE NETWORK ##
        lossMIN = 100000
        flag = 0
        count_change_loss = 0

        for epochID in range(self.numb_epoch):
            print(f"Training {epochID}/{self.numb_epoch-1}")

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            # Update learning rate at each epoch
            # self.adjust_learning_rate(epochID)

            lossTrain = self.train_epoch(
                loss_geb, writer, epochID)

            scene_graphs, issues_1, issues_2 = self.get_val_data()
            with torch.no_grad():
                lossVal = self.validate_retrieval(
                        scene_graphs, issues_1, issues_2)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(lossVal)
            info_txt = f"Epoch {epochID + 1}/{self.numb_epoch} [{timestampEND}]"

            if lossVal < lossMIN:
                count_change_loss = 0
                if lossVal < lossMIN:
                    lossMIN = lossVal
                torch.save({'epoch': epochID,
                            'embed_model_issue_state_dict': self.embed_model_cap.state_dict(),
                            'sent_model_state_dict': self.sent_model.state_dict(),
                            'rels_model_state_dict': self.rels_model.state_dict(),
                            'gcn_model_issue_state_dict': self.gcn_model_cap.state_dict(),
                            'graph_embed_model_state_dict': self.graph_embed_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_loss': lossMIN}, f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}.pth.tar")
                info_txt = info_txt + f" [SAVE]\nLoss Val: {lossVal}"

            else:
                count_change_loss += 1
                info_txt = info_txt + f"\nLoss Val: {lossVal}"
            print(info_txt)
            with open(f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}-REPORT.log", "a") as f_log:
                f_log.write(info_txt)

            writer.add_scalars('Loss Epoch', {'train': lossTrain}, epochID)
            writer.add_scalars('Loss Epoch', {'val': lossVal}, epochID)
            writer.add_scalars('Loss Epoch', {'val-best': lossMIN}, epochID)

            current_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, epochID)

            if count_change_loss >= 15:
                print(
                    f'Early stopping: {count_change_loss} epoch not decrease the loss')
                break

        # f_log.close()
        writer.close()
        torch.save({
            'embed_model_issue_state_dict': self.embed_model_cap.state_dict(),
            'sent_model_state_dict': self.sent_model.state_dict(),
            'rels_model_state_dict': self.rels_model.state_dict(),
            'gcn_model_issue_state_dict': self.gcn_model_cap.state_dict(),
            'graph_embed_model_state_dict': self.graph_embed_model.state_dict()
            }, 'all_pretrained_model.pt')

    # ---------- TRAINING 1 EPOCH ---------
    def train_epoch(self, loss_geb, writer, epochID):
        print(f"Shuffling Training Dataset")
        self.datatrain.create_pairs(seed=1509+epochID+100)
        print(f"Done Shuffling")
        '''
        numb_sample = len(self.datatrain)
        temp = [x['label'] for x in self.datatrain]
        numb_match = np.sum(np.asarray(temp))
        numb_unmatch = numb_sample - numb_match
        print(f"Total Training sample: {numb_sample} --- Matched sample: {numb_match} --- UnMatched sample: {numb_unmatch}")
        '''

        dataloadertrain = make_PairGraphPrecomputeDataLoader(
            self.datatrain, batch_size=self.batch_size, num_workers=0)

        self.embed_model_cap.train()
        self.sent_model.train()
        self.rels_model.train()
        self.gcn_model_cap.train()
        self.graph_embed_model.train()

        loss_report = 0
        count = 0
        numb_iter = len(dataloadertrain)
        print(f"Total iteration: {numb_iter}")
        for batchID, batch in enumerate(dataloadertrain):
            ## CUDA ##
            # if batchID > 2:
            #    break
            issue_1_p_o, issue_1_p_p, issue_1_p_e, issue_1_p_s, issue_1_p_numb_o, issue_1_p_numb_p, issue_1_p_len_p, issue_1_p_len_s, \
                issue_2_p_o, issue_2_p_p, issue_2_p_e, issue_2_p_s, issue_2_p_numb_o, issue_2_p_numb_p, issue_2_p_len_p, issue_2_p_len_s, labels = batch

            issue_1_p_o = issue_1_p_o.to(device)
            issue_1_p_e = issue_1_p_e.to(device)
            issue_2_p_o = issue_2_p_o.to(device)
            issue_2_p_e = issue_2_p_e.to(device)

            # [Caption] Padding
            pad_issue_1_p_s = padding_sequence(
                issue_1_p_s)  # padding sentence
            pad_issue_1_p_p = padding_sequence(
                issue_1_p_p)  # padding predicates
            pad_issue_2_p_s = padding_sequence(
                issue_2_p_s)  # padding sentence
            pad_issue_2_p_p = padding_sequence(
                issue_2_p_p)  # padding predicates

            # Embedding network (object, predicates in image and caption)
            # [Caption] Embed Sentence and Predicates
            eb_pad_issue_1_p_s = self.embed_model_cap(
                pad_issue_1_p_s.to(device))
            eb_pad_issue_1_p_p = self.embed_model_cap(
                pad_issue_1_p_p.to(device))
            eb_pad_issue_2_p_s = self.embed_model_cap(
                pad_issue_2_p_s.to(device))
            eb_pad_issue_2_p_p = self.embed_model_cap(
                pad_issue_2_p_p.to(device))

            # [Caption] Sentence Model
            rnn_eb_pad_issue_1_p_s = self.sent_model(
                eb_pad_issue_1_p_s, issue_1_p_len_s)
            rnn_eb_pad_issue_2_p_s = self.sent_model(
                eb_pad_issue_2_p_s, issue_2_p_len_s)  # ncap, max sent len, dim

            # Concate for batch processing
            rnn_eb_issue_1_p_rels, rnn_eb_issue_1_p_rels_nodes = self.rels_model(
                eb_pad_issue_1_p_p, issue_1_p_len_p)
            rnn_eb_issue_2_p_rels, rnn_eb_issue_2_p_rels_nodes = self.rels_model(
                eb_pad_issue_2_p_p, issue_2_p_len_p)  # total rels, dim

            # [CAPTION] GCN for object and edge in relations
            total_issue_1_p_numb_o = sum(issue_1_p_numb_o)
            total_issue_1_p_numb_p = sum(issue_1_p_numb_p)
            total_issue_2_p_numb_o = sum(issue_2_p_numb_o)
            total_issue_2_p_numb_p = sum(issue_2_p_numb_p)
            eb_issue_1_p_o = torch.zeros(
                total_issue_1_p_numb_o, self.gcn_output_dim).to(device)
            eb_issue_1_p_p = torch.zeros(
                total_issue_1_p_numb_p, self.gcn_output_dim).to(device)
            eb_issue_2_p_o = torch.zeros(
                total_issue_2_p_numb_o, self.gcn_output_dim).to(device)
            eb_issue_2_p_p = torch.zeros(
                total_issue_2_p_numb_p, self.gcn_output_dim).to(device)

            for idx in range(len(rnn_eb_issue_1_p_rels_nodes)):
                edge_1 = issue_1_p_e[idx]  # subject, object

                # <start> is 1st token
                eb_issue_1_p_o[edge_1[0]
                               ] = rnn_eb_issue_1_p_rels_nodes[idx, 1, :]
                eb_issue_1_p_o[edge_1[1]] = rnn_eb_issue_1_p_rels_nodes[idx,
                                                                        issue_1_p_len_p[idx]-2, :]  # <end> is last token
                eb_issue_1_p_p[idx] = torch.mean(
                    rnn_eb_issue_1_p_rels_nodes[idx, 2:(issue_1_p_len_p[idx]-2), :], dim=0)

                eb_issue_1_p_o[edge_1[0]
                               ] = rnn_eb_issue_1_p_rels_nodes[idx, 1, :]
                eb_issue_1_p_o[edge_1[1]] = rnn_eb_issue_1_p_rels_nodes[idx,
                                                                        issue_1_p_len_p[idx]-2, :]  # <end> is last token
                eb_issue_1_p_p[idx] = torch.mean(
                    rnn_eb_issue_1_p_rels_nodes[idx, 2:(issue_1_p_len_p[idx]-2), :], dim=0)

            for idx in range(len(rnn_eb_issue_2_p_rels_nodes)):
                # <start> is 1st token
                edge_2 = issue_2_p_e[idx]
                eb_issue_2_p_o[edge_2[0]
                               ] = rnn_eb_issue_2_p_rels_nodes[idx, 1, :]
                eb_issue_2_p_o[edge_2[1]] = rnn_eb_issue_2_p_rels_nodes[idx,
                                                                        issue_2_p_len_p[idx]-2, :]  # <end> is last token
                eb_issue_2_p_p[idx] = torch.mean(
                    rnn_eb_issue_2_p_rels_nodes[idx, 2:(issue_2_p_len_p[idx]-2), :], dim=0)

                eb_issue_2_p_o[edge_1[0]
                               ] = rnn_eb_issue_2_p_rels_nodes[idx, 1, :]
                eb_issue_2_p_o[edge_1[1]] = rnn_eb_issue_2_p_rels_nodes[idx,
                                                                        issue_2_p_len_p[idx]-2, :]  # <end> is last token
                eb_issue_2_p_p[idx] = torch.mean(
                    rnn_eb_issue_2_p_rels_nodes[idx, 2:(issue_2_p_len_p[idx]-2), :], dim=0)

            eb_issue_1_p_o, eb_issue_1_p_p = self.gcn_model_cap(
                eb_issue_1_p_o, eb_issue_1_p_p, issue_1_p_e)
            eb_issue_2_p_o, eb_issue_2_p_p = self.gcn_model_cap(
                eb_issue_2_p_o, eb_issue_2_p_p, issue_2_p_e)

            # [GRAPHEMB]
            issue_1_geb = self.graph_embed_model(
                eb_issue_1_p_o, eb_issue_1_p_p, issue_1_p_numb_o, issue_1_p_numb_p)  # n_cap, dim
            issue_2_geb = self.graph_embed_model(
                eb_issue_2_p_o, eb_issue_2_p_p, issue_2_p_numb_o, issue_2_p_numb_p)  # n_cap, dim

            # LOSS
            lossvalue = loss_geb(issue_1_geb, issue_2_geb, labels)

            ## Update ##
            self.optimizer.zero_grad()
            lossvalue.backward()
            if self.grad_clip > 0:
                clip_grad_norm(self.params,
                               self.grad_clip)
            self.optimizer.step()
            loss_report += lossvalue.item()
            count += 1
            print(f"Batch Idx: {batchID+1} / {len(dataloadertrain)}: Loss Train {round(loss_report/count, 6)}")
            writer.add_scalars('Loss Training Iter', {
                                'loss': loss_report/count}, epochID * np.floor(numb_iter/20) + np.floor((batchID+1)/20))
        return loss_report/count

    def encode_scene_graphs(self, scene_graphs, batch_size=1):
        issue_dts = IssueDataset(scene_graphs=scene_graphs,
                                   word2index=word2index, numb_sample=None)
        issue_dtld = make_IssueDataLoader(
            issue_dts, batch_size=batch_size, num_workers=0)

        eb_issue_rels_all = []
        eb_issue_sent_all = []
        issue_numb_rels_all = []
        issue_len_sent_all = []
        issue_geb_all = []

        self.embed_model_cap.eval()
        self.gcn_model_cap.eval()
        self.sent_model.eval()
        self.rels_model.eval()
        self.graph_embed_model.eval()

        with torch.no_grad():
            print('Embedding data ...')
            for batchID, batch in enumerate(issue_dtld):
                issue_o, issue_p, issue_e, issue_s, issue_numb_o, issue_numb_p, issue_len_p, issue_len_s = batch
                batch_size = len(issue_numb_o)

                pad_issue_s_concate = padding_sequence(
                    issue_s).to(device)  # padding Sentence
                pad_issue_p_concate = padding_sequence(
                    issue_p).to(device)  # padding Rels

                # Embedding network (object, predicates in image and caption)
                # [Caption] Embed Sentence
                eb_pad_issue_s_concate = self.embed_model_cap(
                    pad_issue_s_concate)
                eb_pad_issue_p_concate = self.embed_model_cap(
                    pad_issue_p_concate)

                # [Caption] Sentence Model
                rnn_eb_pad_issue_s_concate = self.sent_model(
                    eb_pad_issue_s_concate, issue_len_s)
                for idx_sent in range(len(issue_len_s)):
                    eb_issue_sent_all.append(
                        rnn_eb_pad_issue_s_concate[idx_sent, 0:issue_len_s[idx_sent], :].data.cpu())

                rnn_eb_issue_rels, rnn_eb_issue_rels_nodes = self.rels_model(
                    eb_pad_issue_p_concate, issue_len_p)

                # [CAPTION] GCN for object and edge in relations
                total_issue_numb_o = sum(issue_numb_o)
                total_issue_numb_p = sum(issue_numb_p)
                eb_issue_o = torch.zeros(
                    total_issue_numb_o, self.gcn_output_dim).to(device)
                eb_issue_p = torch.zeros(
                    total_issue_numb_p, self.gcn_output_dim).to(device)
                for idx in range(len(rnn_eb_issue_rels_nodes)):
                    edge = issue_e[idx]  # subject, object
                    # <start> is 1st token
                    eb_issue_o[edge[0]
                                 ] = rnn_eb_issue_rels_nodes[idx, 1, :]
                    eb_issue_o[edge[1]] = rnn_eb_issue_rels_nodes[idx,
                                                                      issue_len_p[idx]-2, :]  # <end> is last token
                    eb_issue_p[idx] = torch.mean(
                        rnn_eb_issue_rels_nodes[idx, 2:(issue_len_p[idx]-2), :], dim=0)
                eb_issue_o, eb_issue_p = self.gcn_model_cap(
                    eb_issue_o, eb_issue_p, issue_e)

                # [GRAPHEMB]
                issue_geb = self.graph_embed_model(
                    eb_issue_o, eb_issue_p, issue_numb_o, issue_numb_p)  # n_cap, dim
                issue_geb_all.append(issue_geb)

                pred_offset = 0
                for idx_cap in range(len(issue_numb_p)):
                    eb_issue_rels_all.append(rnn_eb_issue_rels[pred_offset: (
                        pred_offset+issue_numb_p[idx_cap]), :].data.cpu())
                    pred_offset += issue_numb_p[idx_cap]

                issue_numb_rels_all += issue_numb_p
                # issue_len_s is a list already # list [number of caption]
                issue_len_sent_all += issue_len_s

            issue_geb_all = torch.cat(
                issue_geb_all, dim=0).data.cpu().numpy()

        return issue_geb_all

    # ---------- VALIDATE ---------
    def validate_retrieval(self, scene_graphs, issues_1, issues_2):
        print('---------- VALIDATE RETRIEVAL ----------')
        geb_all = self.encode_scene_graphs(
            scene_graphs, batch_size=64)

        print('Scoring ...')
        issue_1_geb_all = geb_all[issues_1]
        issue_2_geb_all = geb_all[issues_2]
        with torch.no_grad():
            score = evalrank(issue_1_geb=issue_1_geb_all, issue_2_geb=issue_2_geb_all)
        return score

# ----- EVALUATOR -----


class Evaluator():
    def __init__(self, info_dict):
        super(Evaluator, self).__init__()
        ##### INIT #####
        self.info_dict = info_dict
        # 50000 - number of training sample in 1 epoch
        self.numb_sample = info_dict['numb_sample']
        self.numb_epoch = info_dict['numb_epoch']  # 10 - number of epoch
        self.unit_dim = 300
        # number of gin layer in graph embedding
        self.numb_gcn_layers = info_dict['numb_gcn_layers']
        # hidden layer in each gin layer
        self.gcn_hidden_dim = info_dict['gcn_hidden_dim']
        self.gcn_output_dim = info_dict['gcn_output_dim']
        self.gcn_input_dim = info_dict['gcn_input_dim']
        self.activate_fn = info_dict['activate_fn']
        self.grad_clip = info_dict['grad_clip']
        self.use_residual = False  # info_dict['use_residual']
        self.batchnorm = info_dict['batchnorm']
        self.dropout = info_dict['dropout']
        self.batch_size = info_dict['batch_size']
        self.save_dir = info_dict['save_dir']
        self.optimizer_choice = info_dict['optimizer']
        self.learning_rate = info_dict['learning_rate']
        self.model_name = info_dict['model_name']
        self.checkpoint = info_dict['checkpoint']
        self.margin_matrix_loss = info_dict['margin_matrix_loss']
        self.rnn_numb_layers = info_dict['rnn_numb_layers']
        self.rnn_bidirectional = info_dict['rnn_bidirectional']
        self.rnn_structure = info_dict['rnn_structure']
        self.visual_ft_dim = info_dict['visual_ft_dim']
        self.ge_dim = info_dict['graph_emb_dim']
        self.include_pred_ft = info_dict['include_pred_ft']
        self.freeze = info_dict['freeze']
        self.datatrain = PairGraphPrecomputeDataset(scene_graphs=train_scene_graphs,
                                                    word2index=word2index,
                                                    train_links=train_links,
                                                    numb_sample=self.numb_sample)

        # DECLARE MODEL
        self.gcn_model_cap = md.GCN_Network(gcn_input_dim=self.gcn_output_dim, gcn_pred_dim=self.gcn_output_dim,
                                            gcn_output_dim=self.gcn_output_dim, gcn_hidden_dim=self.gcn_hidden_dim,
                                            numb_gcn_layers=self.numb_gcn_layers, batchnorm=self.batchnorm,
                                            dropout=self.dropout, activate_fn=self.activate_fn, use_residual=False)

        self.embed_model_cap = md.WordEmbedding(numb_words=TOTAL_WORDS, embed_dim=self.unit_dim,
                                                sparse=False)

        self.sent_model = md.SentenceModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim,
                                           numb_layers=self.rnn_numb_layers,
                                           dropout=self.dropout, bidirectional=self.rnn_bidirectional,
                                           structure=self.rnn_structure)

        self.rels_model = md.RelsModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim,
                                       numb_layers=self.rnn_numb_layers,
                                       dropout=self.dropout, bidirectional=self.rnn_bidirectional,
                                       structure=self.rnn_structure)

        self.graph_embed_model = md.GraphEmb(node_dim=self.gcn_output_dim, edge_dim=self.gcn_output_dim,
                                             fusion_dim=self.ge_dim, activate_fn=self.activate_fn,
                                             batchnorm=self.batchnorm, dropout=self.dropout)

        self.embed_model_cap = self.embed_model_cap.to(device)
        self.sent_model = self.sent_model.to(device)
        self.rels_model = self.rels_model.to(device)
        self.gcn_model_cap = self.gcn_model_cap.to(device)
        self.graph_embed_model = self.graph_embed_model.to(device)

        self.embed_model_cap.eval()
        self.sent_model.eval()
        self.rels_model.eval()
        self.gcn_model_cap.eval()
        self.graph_embed_model.eval()

    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        # ---- Load checkpoint
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.embed_model_cap.load_state_dict(
                modelCheckpoint['embed_model_issue_state_dict'])
            self.sent_model.load_state_dict(
                modelCheckpoint['sent_model_state_dict'])
            self.rels_model.load_state_dict(
                modelCheckpoint['rels_model_state_dict'])
            self.gcn_model_cap.load_state_dict(
                modelCheckpoint['gcn_model_issue_state_dict'])
            self.graph_embed_model.load_state_dict(
                modelCheckpoint['graph_embed_model_state_dict'])
        else:
            print("TRAIN FROM SCRATCH")

    def encode_scene_graphs(self, scene_graphs, batch_size=1):
        issue_dts = IssueDataset(scene_graphs=scene_graphs,
                                 word2index=word2index, numb_sample=None)
        issue_dtld = make_IssueDataLoader(
            issue_dts, batch_size=batch_size, num_workers=0)

        eb_issue_rels_all = []
        eb_issue_sent_all = []
        issue_numb_rels_all = []
        issue_len_sent_all = []
        issue_geb_all = []

        self.embed_model_cap.eval()
        self.gcn_model_cap.eval()
        self.sent_model.eval()
        self.rels_model.eval()
        self.graph_embed_model.eval()

        with torch.no_grad():
            print('Embedding data ...')
            for batchID, batch in enumerate(issue_dtld):
                issue_o, issue_p, issue_e, issue_s, issue_numb_o, issue_numb_p, issue_len_p, issue_len_s = batch
                batch_size = len(issue_numb_o)

                pad_issue_s_concate = padding_sequence(
                    issue_s).to(device)  # padding Sentence
                pad_issue_p_concate = padding_sequence(
                    issue_p).to(device)  # padding Rels

                # Embedding network (object, predicates in image and caption)
                # [Caption] Embed Sentence
                eb_pad_issue_s_concate = self.embed_model_cap(
                    pad_issue_s_concate)
                eb_pad_issue_p_concate = self.embed_model_cap(
                    pad_issue_p_concate)

                # [Caption] Sentence Model
                rnn_eb_pad_issue_s_concate = self.sent_model(
                    eb_pad_issue_s_concate, issue_len_s)
                for idx_sent in range(len(issue_len_s)):
                    eb_issue_sent_all.append(
                        rnn_eb_pad_issue_s_concate[idx_sent, 0:issue_len_s[idx_sent], :].data.cpu())

                rnn_eb_issue_rels, rnn_eb_issue_rels_nodes = self.rels_model(
                    eb_pad_issue_p_concate, issue_len_p)

                # [CAPTION] GCN for object and edge in relations
                total_issue_numb_o = sum(issue_numb_o)
                total_issue_numb_p = sum(issue_numb_p)
                eb_issue_o = torch.zeros(
                    total_issue_numb_o, self.gcn_output_dim).to(device)
                eb_issue_p = torch.zeros(
                    total_issue_numb_p, self.gcn_output_dim).to(device)
                for idx in range(len(rnn_eb_issue_rels_nodes)):
                    edge = issue_e[idx]  # subject, object
                    # <start> is 1st token
                    eb_issue_o[edge[0]
                                 ] = rnn_eb_issue_rels_nodes[idx, 1, :]
                    eb_issue_o[edge[1]] = rnn_eb_issue_rels_nodes[idx,
                                                                      issue_len_p[idx]-2, :]  # <end> is last token
                    eb_issue_p[idx] = torch.mean(
                        rnn_eb_issue_rels_nodes[idx, 2:(issue_len_p[idx]-2), :], dim=0)
                    # if issue_p_len_p[idx] > 5: # pred is longer than 1 words
                    #    eb_issue_p_p[idx] = torch.mean(rnn_eb_issue_p_rels_nodes[idx,2:(issue_p_len_p[idx]-2),:], dim=0)
                    # else:
                    #    eb_issue_p_p[idx] = rnn_eb_issue_p_rels_nodes[idx,2,:]
                eb_issue_o, eb_issue_p = self.gcn_model_cap(
                    eb_issue_o, eb_issue_p, issue_e)

                # [GRAPHEMB]
                issue_geb = self.graph_embed_model(
                    eb_issue_o, eb_issue_p, issue_numb_o, issue_numb_p)  # n_cap, dim
                issue_geb_all.append(issue_geb)

                pred_offset = 0
                for idx_cap in range(len(issue_numb_p)):
                    eb_issue_rels_all.append(rnn_eb_issue_rels[pred_offset: (
                        pred_offset+issue_numb_p[idx_cap]), :].data.cpu())
                    pred_offset += issue_numb_p[idx_cap]

                issue_numb_rels_all += issue_numb_p
                # issue_len_s is a list already # list [number of caption]
                issue_len_sent_all += issue_len_s

            issue_geb_all = torch.cat(
                issue_geb_all, dim=0).data.cpu().numpy()
        return issue_geb_all

    # ---------- VALIDATE ---------
    def validate_retrieval(self, scene_graphs, issues_1, issues_2):
        print('---------- VALIDATE RETRIEVAL ----------')
        geb_all = self.encode_scene_graphs(
            scene_graphs, batch_size=64)
        geb_all = torch.tensor(geb_all)
        torch.save(geb_all, 'all_geb.pt')
        print('Scoring ...')
        issue_1_geb_all = geb_all[issues_1]
        issue_2_geb_all = geb_all[issues_2]
        with torch.no_grad():
            score = evalrank(issue_1_geb=issue_1_geb_all, issue_2_geb=issue_2_geb_all)
        return score