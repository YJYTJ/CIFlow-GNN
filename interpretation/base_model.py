# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import math
from mlp import MLP
import numpy as np


EPS = 1e-05
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class CIFlow_base(nn.Module):
    def __init__(self, args, x_dim, num_class):
        super().__init__()
        hidden_size = args.hidden_size
        self.dropout_p = args.dropout
        self.relu = nn.ReLU()
        self.blocks_num = args.blocks_num
        self.layer_num = args.layer_num
        self.fc_out = nn.Sequential(nn.Linear(hidden_size, num_class))
        self.important_c = args.important_c
        self.node_encoder_list = nn.ModuleList()
        for _ in range(self.blocks_num):
            self.node_encoder_list.append(nn.Linear(x_dim ,hidden_size))
        self.concat_mlp = MLP(2, self.layer_num+1 ,self.layer_num+1, 1)
        self.cat = nn.Linear((self.layer_num+1)*hidden_size , hidden_size)
        self.batch_norms = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for _ in range(self.layer_num+1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.lns.append(torch.nn.LayerNorm(hidden_size))
        self.softmax = nn.Softmax(dim=-1)
        self.clusters = args.clusters
        self.S_mlp = MLP(2, hidden_size, math.ceil((hidden_size+args.clusters)/2),args.clusters)
        self.final = nn.Linear(in_features = hidden_size*args.clusters, out_features = 2)
        self.num_graph_filter = args.num_graph_filter
        self.n_classes =  2
        self.correlation_inter = nn.Parameter(F.softmax(torch.rand(size=(self.n_classes, self.num_graph_filter)), dim=0))
        self.correlation_intra = nn.Parameter(F.softmax(torch.rand(size=(self.n_classes, self.num_graph_filter)), dim=1))
        self.sigmoid = nn.Sigmoid()
        self.prototype_features_MLP = MLP(2,hidden_size,8,self.num_graph_filter)

    
    def forward(self,W,data, x,input_adj,H_all,targets, batch,is_test):

        graph_indicator = batch
        unique, counts = torch.unique(graph_indicator, return_counts=True)

        E_lists = []
        
        # Compute soft cluster assignment matrix S
        S_all1 = self.S_mlp(H_all)
        S_all = self.softmax(S_all1)

        Cluster_assign_idx = [[] for i in range(len(unique))]
        count = 0

        # Compute embeddings of the clusters E for each graph
        for i in range(len(unique)):
            ind = (graph_indicator==i).nonzero(as_tuple=True)[0]
            H = H_all[ind]
            S = S_all[ind]
            S_ = torch.argmax(S, dim=1)
            for j in range(self.clusters):
                idx = (S_==j).nonzero(as_tuple=True)[0]
                Cluster_assign_idx[i].append(idx)
                
            E = S.t() @ H
            E_lists.append(E)
            count += len(ind)
        E_all = torch.stack(E_lists)

        E_list = E_all.reshape(E_all.shape[0],-1)
        
        
        # Prediction y1
        pred1 = self.final(E_list)    
        pred1 = F.dropout(pred1,self.dropout_p,training = self.training)

        # Reparameterization trick
        pred_softmax = torch.softmax(pred1, dim=-1)
        with torch.no_grad():
            sample_cat = torch.multinomial(pred_softmax, 1, replacement=False).flatten().to(device)
            ind_positive_sample = sample_cat == targets.squeeze()  # mark wrong sample results
            sample_cat_oh = F.one_hot(sample_cat, num_classes=pred1.shape[1]).float().to(device)
            epsilon = torch.where(sample_cat_oh != 0, 1 - pred_softmax, -pred_softmax).detach()
        sample_cat_oh = pred_softmax + epsilon

        
        Q = self.prototype_features_MLP(E_all)
        Q = F.softmax(Q,dim=-1)

        correlation_inter = F.softmax(self.correlation_inter, dim=0)
        correlation_intra = F.softmax(self.correlation_intra, dim=1)
        correlation = torch.mul(correlation_inter,correlation_intra)

        correlation_mask = sample_cat_oh @ correlation

        correlation_mask = correlation_mask.unsqueeze(1).repeat(1,self.clusters,1) #(128,2,4)

        prototype_mask = torch.mul(Q,correlation_mask)#(128,2,4)
        feature_mask = torch.sum(prototype_mask,dim=-1)

        E_sorted_list_mask = torch.einsum('bi,bij->bij', feature_mask, E_all)
        E_sorted_list_mask1 = E_sorted_list_mask.reshape(E_sorted_list_mask.shape[0],-1)

        pred2 = self.final(E_sorted_list_mask1)

        pred2 = F.dropout(pred2,self.dropout_p,training = self.training)
        if is_test==True:
            with torch.no_grad():
                important_nodes_set, sparsity, important_nodes = self.replace_top_c_with_one(Cluster_assign_idx,counts, feature_mask,self.clusters, c=self.important_c)
                
        else:
            important_nodes = None

        return pred1, pred2, ind_positive_sample, important_nodes, S_all, E_all, Q


    
    def get_emb(self, W,features):
        """
        Compute cluster-friendly node embeddings
        """
        W_list = []
        W_list.append(W)
        for p in range(1, self.layer_num):
            W_list.append(torch.sparse.mm(W,W_list[-1]))
        node_feature_list = []
        hidden_rep_tmp = []
        for i in range(self.blocks_num):
            node_feature_tmp = self.node_encoder_list[i](features)
            node_feature_list.append(node_feature_tmp)
        node_feature = torch.stack(node_feature_list,0)
        node_feature = torch.mean(node_feature,0)
        hidden_rep = []
        hidden_rep.append(node_feature)
        for p in range(self.layer_num):
            hidden_rep_tmp = []
            for b in range(self.blocks_num):
                H = W_list[p] @ node_feature_list[b]
                H = self.batch_norms[p](H)
                H = F.relu(H)
                hidden_rep_tmp.append(H)
            node_feature = torch.stack(hidden_rep_tmp,0)
            node_feature = torch.mean(node_feature,0)
            hidden_rep.append(node_feature)  
        n,dim = H.shape
        H_all = torch.stack(hidden_rep,0)
        H_all = torch.reshape(H_all,(self.layer_num+1,-1))
        H_all = H_all.t()
        H_all = self.concat_mlp(H_all) 
        H_all = H_all.squeeze()
        H_all = torch.reshape(H_all,(n,dim))
        H_all = self.batch_norms[-1](H_all)
        H_all = F.relu(H_all)
        return H_all



    def replace_top_c_with_one(self,Cluster_assign_idx,counts,matrix,cluster_num, c=1):
        # 获取每一行的前c大值的索引
        _, top_c_indices = torch.topk(matrix, c, dim=1)
        
        # 创建一个与原矩阵形状相同的零张量
        result = torch.zeros_like(matrix).to(device)
        
        # 将每一行的前c大值位置设置为1
        result.scatter_(1, top_c_indices, 1)

        important_nodes = []
        important_nodes_ori = []
        add_count = 0
        sparsity = 0
        
        for i in range(result.shape[0]):
            important_nodes_per_graph = []
            important_nodes_per_graph_ori = []
            c1=c
            while(True):
                breakFlag = 0
                for j in range(result.shape[1]):
                    if result[i][j] and len(Cluster_assign_idx[i][j])>0:
                        
                        important_nodes_per_graph.extend(Cluster_assign_idx[i][j]+add_count)
                        important_nodes_per_graph_ori.extend(Cluster_assign_idx[i][j])
                        breakFlag = breakFlag +1
                if breakFlag == c:
                    break
                if c1 == cluster_num:
                    break
                c1 = c1+1
                important_nodes_per_graph = []
                important_nodes_per_graph_ori = []
                _, top_c_indices = torch.topk(matrix[i], c1)
                result[i][top_c_indices] = 1

                    
            important_nodes.extend(important_nodes_per_graph)
            important_nodes_ori.append(torch.tensor(important_nodes_per_graph_ori))
            sparsity -= len(important_nodes_per_graph)/counts[i]
            add_count = add_count+counts[i]
        
        important_nodes = torch.tensor(important_nodes)
        important_nodes = set(np.array(important_nodes.detach().cpu()))
        return important_nodes,sparsity,important_nodes_ori
    

    