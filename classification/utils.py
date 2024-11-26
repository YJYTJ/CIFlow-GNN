import os
import sys
import torch
import torch.utils.data as utils
import numpy as np
import logging
from math import ceil
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import OneHotEncoder,normalize
import networkx as nx
from functools import cmp_to_key
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_data(ds_name, use_node_labels=False,use_node_attri=False):    
    graph_indicator = np.loadtxt("datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name), dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)
    
    edges = np.loadtxt("datasets/%s/%s_A.txt"%(ds_name,ds_name), dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
    
    if use_node_labels:
        x = np.loadtxt("datasets/%s/%s_node_labels.txt"%(ds_name,ds_name), dtype=np.int64).reshape(-1,1)
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)
        # x_label = np.argmax(x,axis=1)
        if use_node_attri:
            x1 = np.loadtxt("datasets/%s/%s_node_attributes.txt"%(ds_name,ds_name), delimiter=',',dtype=np.float64)#.reshape(-1,1)
            x = np.concatenate((x,x1),1)
    else:
        x = A.sum(axis=1)
        
    adj = []
    features = []
 
    # feature_labels = []
    idx = 0
    for i in range(graph_size.size):     
        adj.append(A[idx:idx+graph_size[i],idx:idx+graph_size[i]])
        features.append(x[idx:idx+graph_size[i],:])
        idx += graph_size[i]

    class_labels = np.loadtxt("datasets/%s/%s_graph_labels.txt"%(ds_name,ds_name), dtype=np.int64)
    return adj, features, class_labels 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_batches_trans(dataloader):
    datas = []
    for one_dataloader in [dataloader['train'], dataloader['eval'], dataloader['test']]:
        adj_lst = list()
        features_lst = list()
        graph_indicator_lst = list()
        y_lst = list() 
        
        for batch in one_dataloader:
            adj_batch = torch.zeros((batch.num_nodes, batch.num_nodes), dtype=torch.float)
            adj_batch[batch.edge_index[0], batch.edge_index[1]] = 1
            # print(batch.edge_index[0])
            # print(batch.edge_index[1])
            # print(adj_batch)
            # adj_batch[batch.edge_index[1], batch.edge_index[0]] = 1
            features_batch = batch.x  # 假设节点特征存储在 x 属性中
            y_batch = batch.y     
            graph_indicator_batch = torch.zeros(features_batch.shape[0],dtype=torch.long)
            idx = 0
            for j,Data in enumerate(batch.to_data_list()):
                nodes_num = Data.num_nodes
                graph_indicator_batch[idx:idx+nodes_num] = j
                idx += nodes_num
        
            adj_lst.append(adj_batch.to(device))
            features_lst.append(torch.FloatTensor(features_batch).to(device))
            graph_indicator_lst.append(graph_indicator_batch.to(device))
            y_lst.append(torch.LongTensor(y_batch).to(device))
            # adj_lst.append(adj_batch)
            # features_lst.append(torch.FloatTensor(features_batch))
            # graph_indicator_lst.append(graph_indicator_batch)
            # y_lst.append(torch.LongTensor(y_batch))
        datas.append([adj_lst, features_lst, graph_indicator_lst, y_lst])
        
        
        
        
    

    return datas

def generate_batches( adj, features, y, batch_size, device, shuffle=False):
    N = len(y)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    n_batches = ceil(N/batch_size)

    adj_lst = list()
    features_lst = list()
    graph_indicator_lst = list()
    y_lst = list() 
    centrality_lst = list()
    for i in range(0, N, batch_size):
        n_graphs = min(i+batch_size, N) - i
        n_nodes = sum([adj[index[j]].shape[0] for j in range(i, min(i+batch_size, N))])

        adj_batch = lil_matrix((n_nodes, n_nodes))
        features_batch = np.zeros((n_nodes, features[0].shape[1]))
        graph_indicator_batch = np.zeros(n_nodes)
        y_batch = np.zeros(n_graphs)


        idx = 0
        for j in range(i, min(i+batch_size, N)):
            n = adj[index[j]].shape[0]
            adj_batch[idx:idx+n, idx:idx+n] = adj[index[j]]    
            features_batch[idx:idx+n,:] = features[index[j]]
            graph_indicator_batch[idx:idx+n] = j-i
            y_batch[j-i] = y[index[j]]
            idx += n
                  
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
        features_lst.append(torch.FloatTensor(features_batch).to(device))
        graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
        y_lst.append(torch.LongTensor(y_batch).to(device))


    return adj_lst, features_lst, graph_indicator_lst, y_lst

def normalized_laplacian(adj_matrix):
    I = torch.eye(adj_matrix.shape[0]).to(device)
    adj_matrix = adj_matrix + I
    R = torch.sum(adj_matrix, dim=1)
    R_sqrt = 1/torch.sqrt(R)
    D_sqrt = torch.diag(R_sqrt)
    # I = np.eye(adj_matrix.shape[0])
    return torch.matmul(torch.matmul(D_sqrt, adj_matrix), D_sqrt)



def normalized_laplacian_W_batch(adj_matrix_batch,power_num):
    L_batch = []
    W_batch = []
    for i in range(len(adj_matrix_batch)):
        adj_matrix = adj_matrix_batch[i].to_dense()
        I = torch.eye(adj_matrix.shape[0]).to(device)
        adj_matrix = adj_matrix + I
        D = torch.sum(adj_matrix, dim=1)
        R_sqrt = 1/torch.sqrt(D)
        D_sqrt = torch.diag(R_sqrt)
        L = torch.diag(D) - adj_matrix
        D_sqrt_sp = D_sqrt.to_sparse()
        L_sp = L.to_sparse()
        L_sym = torch.sparse.mm(torch.sparse.mm(D_sqrt_sp, L_sp), D_sqrt_sp)
        # adj_batch_sp = adj_matrix.to_sparse()
        adj_batch_sp = adj_matrix_batch[i]
        W = torch.sparse.mm(torch.sparse.mm(D_sqrt_sp, adj_batch_sp), D_sqrt_sp)
        L_batch.append(L_sym)
        W_batch.append(W)
    return L_batch,W_batch



