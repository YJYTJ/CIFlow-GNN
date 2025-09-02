import sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from src.utils import get_preds, generate_A_L_W
import torch.nn.functional as F
from draw import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def eval_one_batch(A, L, W, CIFlow_model, data, is_test):
    
    CIFlow_model.base.eval()
    loss, loss_dict, pred1, pred2, important_nodes = CIFlow_model.forward(is_test,A,L,W, data)
    return loss_dict, pred1.data.cpu(), pred2.data.cpu(), important_nodes
      


def train_one_batch(A,L,W, CIFlow_model, data,is_test):

    CIFlow_model.base.train()
    loss, loss_dict, pred1, pred2, important_nodes = CIFlow_model.forward(is_test,A,L, W, data)
    CIFlow_model.optimizer.zero_grad()
    loss.backward()
    CIFlow_model.optimizer.step()
    return loss_dict, pred1.data.cpu(),pred2.data.cpu(),important_nodes




def important_portion_func(dataset_name, data, important_nodes):
    graph_indicator = data.batch
    unique, counts = torch.unique(graph_indicator, return_counts=True)
    count  = 0
    auc_list = []
    if dataset_name == 'solubility' or dataset_name == 'benzene':
        important_label = 1
    elif dataset_name == 'mutagenicity' or dataset_name == 'labeled-motifs':
        important_label = 0
    for i in range(len(unique)):
        ind = (graph_indicator==i).nonzero(as_tuple=True)[0]
        if data.y[i] == important_label:
            important_nodes_graphi = important_nodes[i].cpu().numpy()
            ground_truth_graphi = data.ground_truth[count:count+len(ind)]
            ground_truth_arrayi = ground_truth_graphi.cpu().numpy()
            important_nodes_arrayi = np.zeros(len(ind),dtype=int)
            if len(important_nodes_graphi)>0:
                important_nodes_arrayi[important_nodes_graphi] = 1
            auc = roc_auc_score(ground_truth_arrayi,important_nodes_arrayi)
            auc_list.append(auc)
        count = count+len(ind)
    return auc_list

     

def run_one_epoch(A_list,L_list, W_list, CIFlow_model, data_loader, epoch, phase, dataset_name,is_test, is_train):
    loader_len = len(data_loader)
    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

    all_loss_dict = {}
    all_clf_labels, all_clf_logits1,all_clf_logits2 = ([] for i in range(3))
    pred1_list = []
    targets_list = []
    ratio_list= []

    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        if is_train == True:
            A, L, W = generate_A_L_W(data)
        else:
            A, L, W = A_list[idx].to(device), L_list[idx],W_list[idx]

         
     
        loss_dict, pred1, pred2, important_nodes = run_one_batch(A,L, W, CIFlow_model, data.to(device), is_test)
        
        desc, _, _, _ = log_epoch(epoch, phase, loss_dict, data.y.data.cpu(), pred1,pred2,
                                    dataset_name, batch=True)
   
        if is_test == True:
            pred1_list.append(pred1)      
            ratio_list.extend(important_portion_func(dataset_name,data, important_nodes))

        for k, v in loss_dict.items():
            all_loss_dict[k] = all_loss_dict.get(k, 0) + v
       
        all_clf_labels.append(data.y.data.cpu()), all_clf_logits1.append(pred1)
        all_clf_logits2.append(pred2)

        if idx == loader_len - 1:
            all_clf_labels, all_clf_logits1, all_clf_logits2 = torch.cat(all_clf_labels), torch.cat(all_clf_logits1), torch.cat(all_clf_logits2)
            
            for k, v in all_loss_dict.items():
                all_loss_dict[k] = v / loader_len
            desc, clf_acc1, clf_acc2, loss = log_epoch(epoch, phase, all_loss_dict, all_clf_labels.squeeze(),
                                                                     all_clf_logits1,all_clf_logits2,
                                                                    dataset_name, batch=False)
        pbar.set_description(desc)
    
    if is_test==True:
        ratio_list = torch.tensor(ratio_list)
        auc = torch.mean(ratio_list).detach().cpu()
        print(f'[Epoch: {epoch}]: CIFlow_{phase} auc: {auc:.3f}' )
        return loss, auc

    else:
        return loss







def log_epoch(epoch, phase, loss_dict, clf_labels, pred1,pred2, dataset_name, batch):
    desc = f'[Epoch: {epoch}]: CIFlow_{phase}........., ' if batch else f'[Epoch: {epoch}]: CIFlow_{phase} finished, '
    for k, v in loss_dict.items():
        desc += f'{k}: {v:.3f}, '

    eval_desc, clf_acc1,clf_acc2 = get_eval_score( clf_labels, pred1,pred2, dataset_name, batch)
    desc += eval_desc
    return desc, clf_acc1,clf_acc2, loss_dict['loss']


def get_eval_score( clf_labels, pred1,pred2, dataset_name, batch):
    
    
    clf_preds1 = get_preds(pred1)
    clf_acc1 = (clf_preds1 == clf_labels.squeeze()).sum().item() / clf_labels.shape[0]
    clf_preds2 = get_preds(pred2)
    clf_acc2 = (clf_preds2 == clf_labels.squeeze()).sum().item() / clf_labels.shape[0]

    if batch:
        return f'acc: {clf_acc1:.3f}', None, None
 
    desc = f'acc: {clf_acc1:.3f}'
    return desc, clf_acc1,clf_acc2

