import sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from src.utils import get_preds
import torch.nn.functional as F
from draw import *
from sklearn.metrics import roc_auc_score


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def eval_one_batch(A,L,W, CIFlow_model, data,is_test):

    CIFlow_model.base.eval()

    loss,loss_dict, pred1,pred2,pred_fidelity_m,pred_fidelity_1_m,sparsity,important_nodes = CIFlow_model.forward(is_test,A,L,W, data)
  
    return loss_dict, pred1.data.cpu(),pred2.data.cpu(),pred_fidelity_m,pred_fidelity_1_m,sparsity,important_nodes
      


def train_one_batch(A,L,W, CIFlow_model, data,is_test):

    CIFlow_model.base.train()
    loss,loss_dict, pred1,pred2,pred_fidelity_m,pred_fidelity_1_m,sparsity,important_nodes = CIFlow_model.forward(is_test,A,L, W, data)
    CIFlow_model.optimizer.zero_grad()
    loss.backward()
    CIFlow_model.optimizer.step()
    
    return loss_dict, pred1.data.cpu(),pred2.data.cpu(),pred_fidelity_m,pred_fidelity_1_m,sparsity,important_nodes


def generate_A_L_W(batch):

    adj_batch_ad = torch.zeros((batch.num_nodes+1, batch.num_nodes+1), dtype=torch.float)
    adj_batch_ad[batch.edge_index[0], batch.edge_index[1]] = 1
    adj_batch_ad[batch.edge_index[1], batch.edge_index[0]] = 1
    adj_batch = adj_batch_ad[:batch.num_nodes][:,:batch.num_nodes]
    A = adj_batch_ad.to(device)
    I = torch.eye(adj_batch.shape[0])
    adj_batch = adj_batch + I
    adj_batch = adj_batch.to(device)
    D = torch.sum(adj_batch, dim=1)
    R_sqrt = 1/torch.sqrt(D)
    D_sqrt = torch.diag(R_sqrt)
    L = torch.diag(D) - adj_batch
    D_sqrt_sp = D_sqrt.to_sparse()
    L_sp = L.to_sparse()
    L_sym = torch.sparse.mm(torch.sparse.mm(D_sqrt_sp, L_sp), D_sqrt_sp)
    adj_batch_sp = adj_batch.to_sparse()
    W = torch.sparse.mm(torch.sparse.mm(D_sqrt_sp, adj_batch_sp), D_sqrt_sp)

    return A,L_sym, W

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

     

def run_one_epoch(flag,A_list,L_list, W_list, CIFlow_model, data_loader, epoch, phase, dataset_name,is_test):
    loader_len = len(data_loader)
    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

    all_loss_dict = {}
    all_clf_labels, all_clf_logits1,all_clf_logits2 = ([] for i in range(3))
    pred1_list = []
    fidelity_m_list = []
    fidelity_1_m_list = []
    sparsity_list = []
    targets_list = []
    ratio_list= []

    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        if flag==False:
            A, L, W = generate_A_L_W(data)

        else:
            A, L, W = A_list[idx].to(device), L_list[idx],W_list[idx]

         
     
        loss_dict, pred1,pred2,pred_fidelity_m,pred_fidelity_1_m,sparsity,important_nodes = run_one_batch(A,L, W, CIFlow_model, data.to(device),is_test)
        
        desc, _, _, _ = log_epoch(epoch, phase, loss_dict, data.y.data.cpu(), pred1,pred2,
                                    dataset_name, batch=True)
   
        if is_test==True:
            pred1_list.append(pred1)
            fidelity_m_list.append(pred_fidelity_m.data.cpu())
            fidelity_1_m_list.append(pred_fidelity_1_m.data.cpu())
            sparsity_list.append(sparsity)
            targets_list.append(data.y.data.cpu())         
            ratio_list.extend(important_portion_func(dataset_name,data, important_nodes))

  
        for k, v in loss_dict.items():
            all_loss_dict[k] = all_loss_dict.get(k, 0) + v

       
        all_clf_labels.append(data.y.data.cpu()), all_clf_logits1.append(pred1)
        all_clf_logits2.append(pred2)

        if idx == loader_len - 1:
            all_clf_labels, all_clf_logits1 = torch.cat(all_clf_labels), torch.cat(all_clf_logits1)
            all_clf_logits2 = torch.cat(all_clf_logits2)

            for k, v in all_loss_dict.items():
                all_loss_dict[k] = v / loader_len
            desc, clf_acc1,clf_acc2, loss = log_epoch(epoch, phase, all_loss_dict, all_clf_labels.squeeze(),
                                                                     all_clf_logits1,all_clf_logits2,
                                                                    dataset_name, batch=False)
        pbar.set_description(desc)
    
    if is_test==True:
        pred1_list = torch.cat(pred1_list,0)
        fidelity_m_list = torch.cat(fidelity_m_list,0)
        fidelity_1_m_list = torch.cat(fidelity_1_m_list,0)
        targets_list = torch.cat(targets_list,0)
        Fidelity =  Fidelity_func(fidelity_1_m_list,fidelity_m_list,targets_list)
        Sparsity  =  Sparsity_func(sparsity_list,pred1_list.shape[0])
        ratio_list = torch.tensor(ratio_list)
        auc = torch.mean(ratio_list).detach().cpu()
        

        return loss, clf_acc1, auc, Fidelity, Sparsity

    else:
        return loss, clf_acc1, 0, None, None




def Fidelity_func(fidelity_1_m_list,fidelity_m_list,targets_list):
    N = fidelity_1_m_list.shape[0]
    y_hat1 = torch.argmax(fidelity_m_list,dim=1)
    same_elements1 = (y_hat1 == targets_list)
    count1 = same_elements1.sum().item()
    y_hat2 = torch.argmax(fidelity_1_m_list,dim=1)
    same_elements2 = (y_hat2 == targets_list)
    count2 = same_elements2.sum().item()
    Fidelity = (count1-count2)/N
    return Fidelity




def Sparsity_func(sparsity_list,N):
    sp = (torch.sum(torch.tensor(sparsity_list))+N)/N
    return sp 



def log_epoch(epoch, phase, loss_dict, clf_labels, pred1,pred2, dataset_name, batch):
    desc = f'[Epoch: {epoch}]: gsat_{phase}........., ' if batch else f'[Epoch: {epoch}]: gsat_{phase} finished, '
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

