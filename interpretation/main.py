import sys
sys.path.append('../src')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
from model import CIFlow
from base_model import CIFlow_base
from src.utils import get_data_loaders, set_seed
from trainer import run_one_epoch
import argparse
import copy
import numpy as np

def args_parser():  
    parser = argparse.ArgumentParser(description='CIFlow-GNN')
    parser.add_argument('--lambda_con', type=float, default=0.01)
    parser.add_argument('--lambda_fea', type=float, default=0.01)
    parser.add_argument('--lambda_2', type=float, default=0.5)
    parser.add_argument('--lambda_proto', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--clusters', type=int, default=2)
    parser.add_argument('--power_num', type=int, default=1)
    parser.add_argument('--blocks_num', type=int, default=6)
    parser.add_argument('--num_graph_filter', type=int, default=3)
    parser.add_argument('--result_folder',  type=str, default='results', help='results folder')
    parser.add_argument('--important_c', type=int, default=1, help='select top c clusters as important subgraph')
    parser.add_argument('--dataset_name',  type=str, default='labeled-motifs', help='solubility/benzene/mutagenicity/labeled-motifs')
    args = parser.parse_args()
    return args


args = args_parser()
dataset_name = args.dataset_name

model_name= 'CIFlow'
seed = args.seed
set_seed(args.seed)

data_dir = Path('data')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_config = {'model_name': 'GIN', 'hidden_size': 64, 'n_layers': 2, 'dropout_p': 0.3, 'use_edge_attr': True}

    

loaders, test_set, x_dim, num_class = get_data_loaders(data_dir, dataset_name, batch_size=128, random_state=seed,
                                                                                splits={'train': 0.8, 'valid': 0.1, 'test': 0.1}, 
                                                                                mutag_x=True if dataset_name == 'mutagenicity' else False)
  

CIFlow_base_model = CIFlow_base(args, x_dim, num_class, model_config).to(device)
optimizer = torch.optim.Adam(list(CIFlow_base_model.parameters()), lr=1e-3, weight_decay=3.0e-6)
CIFlow_model = CIFlow(args,CIFlow_base_model, optimizer)

def generate_A_L_W_all(dataloader):
    L_list = []
    W_list = []
    A_list = []

    datalodaer_cp = copy.deepcopy(dataloader)

    for i,batch in enumerate(datalodaer_cp):
        adj_batch_ad = torch.zeros((batch.num_nodes+1, batch.num_nodes+1), dtype=torch.float)
        adj_batch_ad[batch.edge_index[0], batch.edge_index[1]] = 1
        adj_batch_ad[batch.edge_index[1], batch.edge_index[0]] = 1
        adj_batch = adj_batch_ad[:batch.num_nodes][:,:batch.num_nodes]
        I = torch.eye(adj_batch.shape[0])
        A_list.append(adj_batch_ad)
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
        L_list.append(L_sym)
        W_list.append(W)


    return A_list,L_list, W_list

A_val,L_val, W_val = generate_A_L_W_all(loaders['valid'])
A_test,L_test, W_test = generate_A_L_W_all(loaders['test'])




ResultFolderPATH = args.result_folder +'/'+dataset_name+ '/'  \
    + 'lambda_con' + str(args.lambda_con)+ '/'  \
    + 'lambda_fea' + str(args.lambda_fea)+ '/'  \
    + 'lambda_proto' + str(args.lambda_proto)+ '/'  \
    + 'lambda_2' + str(args.lambda_2)+ '/'  \
    + 'power_num' + str(args.power_num)+ '/'  \
    + 'blocks_num' + str(args.blocks_num)+ '/' \
    + 'important_c' + str(args.important_c)+ '/'  \
    + 'clusters' + str(args.clusters)+ '/'  \
    + 'num_graph_filter' + str(args.num_graph_filter)+ '/'


if not os.path.exists(ResultFolderPATH):
    os.makedirs(ResultFolderPATH)


acc_per_epoch = np.zeros((args.epochs, 5))
for epoch in range(args.epochs):
    train_res = run_one_epoch(False,None,None,None, CIFlow_model, loaders['train'], epoch, 'train', dataset_name,is_test=False)
    valid_res = run_one_epoch(True,A_val,L_val, W_val, CIFlow_model, loaders['valid'], epoch, 'valid', dataset_name,is_test=False)
    test_res = run_one_epoch(True,A_test,L_test, W_test, CIFlow_model, loaders['test'], epoch, 'test', dataset_name,is_test=True)
    loss, acc, auc, Fidelity, Sparsity = test_res
    acc_per_epoch[epoch][0] = valid_res[0] #val loss
    acc_per_epoch[epoch][1] = acc
    acc_per_epoch[epoch][2] = auc
    acc_per_epoch[epoch][3] = Fidelity
    acc_per_epoch[epoch][4] = Sparsity


    with open(ResultFolderPATH + '/acc_results.txt', 'a+') as f:
        f.write(str(acc_per_epoch[epoch][0]) + ' ')
        f.write(str(acc_per_epoch[epoch][1]) + ' ')
        f.write(str(acc_per_epoch[epoch][2]) + ' ')
        f.write(str(acc_per_epoch[epoch][3]) + ' ')
        f.write(str(acc_per_epoch[epoch][4]) + '\n')


    print('='*50)
    print('='*50)



