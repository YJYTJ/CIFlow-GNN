import sys
sys.path.append('../src')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
from model import CIFlow
from base_model import CIFlow_base
from src.utils import get_data_loaders, set_seed, generate_A_L_W_all
from trainer import run_one_epoch
import argparse
import numpy as np

def args_parser():  
    parser = argparse.ArgumentParser(description='CIFlow-GNN')
    parser.add_argument('--lambda_con', type=float, default=0.05)
    parser.add_argument('--lambda_fea', type=float, default=0.1)
    parser.add_argument('--lambda_2', type=float, default=0.3)
    parser.add_argument('--lambda_proto', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42) # 42 0.818
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--clusters', type=int, default=2)
    parser.add_argument('--layer_num', type=int, default=4)
    parser.add_argument('--blocks_num', type=int, default=1)
    parser.add_argument('--num_graph_filter', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--result_folder',  type=str, default='mini_results', help='results folder')
    parser.add_argument('--important_c', type=int, default=1, help='select top c clusters as important subgraph')
    parser.add_argument('--dataset_name',  type=str, default='labeled-motifs', help='solubility/benzene/mutagenicity/labeled-motifs')
    args = parser.parse_args()
    return args

args = args_parser()
dataset_name = args.dataset_name

model_name= 'CIFlow'
seed = args.seed
set_seed(args.seed)
data_dir = Path('./data')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
# load dataset and data splits
loaders, test_set, x_dim, num_class = get_data_loaders(data_dir, dataset_name, batch_size=128, random_state=seed,
                                                                                splits={'train': 0.8, 'valid': 0.1, 'test': 0.1}, 
                                                                                mutag_x=True if dataset_name == 'mutagenicity' else False)
  
# CIFlow-GNN model
CIFlow_base_model = CIFlow_base(args, x_dim, num_class).to(device)

# optimizer
optimizer = torch.optim.Adam(list(CIFlow_base_model.parameters()), lr=1e-3, weight_decay=3.0e-6)

# loss function
CIFlow_model = CIFlow(args,CIFlow_base_model, optimizer)

# precompute adj matrices, laplacian matrices, and normalized adjacency matrice with self-loops
A_val,L_val, W_val = generate_A_L_W_all(loaders['valid'])
A_test,L_test, W_test = generate_A_L_W_all(loaders['test'])

# record the results
ResultFolderPATH = args.result_folder +'/'+dataset_name+ '/'  \
    + 'lambda_con' + str(args.lambda_con)+ '/'  \
    + 'lambda_fea' + str(args.lambda_fea)+ '/'  \
    + 'lambda_proto' + str(args.lambda_proto)+ '/'  \
    + 'lambda_2' + str(args.lambda_2)+ '/'  \
    + 'layer_num' + str(args.layer_num)+ '/'  \
    + 'blocks_num' + str(args.blocks_num)+ '/' \
    + 'important_c' + str(args.important_c)+ '/'  \
    + 'clusters' + str(args.clusters)+ '/'  \
    + 'num_graph_filter' + str(args.num_graph_filter)+ '/' \
    + 'seed' + str(args.seed)+ '/'
if not os.path.exists(ResultFolderPATH):
    os.makedirs(ResultFolderPATH)


acc_per_epoch = np.zeros((args.epochs, 2))
for epoch in range(args.epochs):
    train_loss = run_one_epoch(None, None, None, CIFlow_model, loaders['train'], epoch, 'train', dataset_name, is_test = False, is_train = True)
    valid_loss = run_one_epoch(A_val, L_val, W_val, CIFlow_model, loaders['valid'], epoch, 'valid', dataset_name, is_test=False, is_train = False)
    test_loss, test_auc = run_one_epoch(A_test, L_test, W_test, CIFlow_model, loaders['test'], epoch, 'test', dataset_name, is_test=True, is_train = False)

    acc_per_epoch[epoch][0] = valid_loss
    acc_per_epoch[epoch][1] = test_auc

    with open(ResultFolderPATH + '/acc_results.txt', 'a+') as f:
        f.write(str(acc_per_epoch[epoch][0]) + ' ')
        f.write(str(acc_per_epoch[epoch][1]) + '\n')

    print('='*50)
    print('='*50)

ind = np.argmin(acc_per_epoch[:,0])
auc = acc_per_epoch[ind,1]
print(f'The final ROC AUC for dataset {dataset_name} is : {auc:.3f}.')

