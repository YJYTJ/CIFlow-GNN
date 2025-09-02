import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
from mlp import MLP
import time
# from utils_thread import normalized_laplacian
import math
import torch.nn.init as init
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CIFlow(nn.Module):
    def __init__(self,args,input_dim, output_dim ):
        
        super(CIFlow, self).__init__()
        self.layer_num = args.layer_num
        self.blocks_num = args.blocks_num
        self.no_norm = args.no_norm
        self.dropout_rate = args.dropout_rate
        self.class_no = output_dim
        self.d = input_dim
        latent_dim1 = args.latent_dim1
        latent_dim2 = args.latent_dim2
        self.K_cluster = args.K_cluster
        self.concat_mlp = MLP(2,self.layer_num+1 , latent_dim2,1 )
        self.S_mlp = MLP(2,latent_dim1 , latent_dim2,args.K_cluster )

        self.final = nn.Linear(in_features = latent_dim1*args.K_cluster , out_features = output_dim)
        self.num_graph_filter = args.num_graph_filter
        self.n_classes =  output_dim 
        self.correlation_inter = nn.Parameter(F.softmax(torch.rand(size=(self.n_classes, self.num_graph_filter)), dim=0))
        self.correlation_intra = nn.Parameter(F.softmax(torch.rand(size=(self.n_classes, self.num_graph_filter)), dim=1))
        self.sigmoid = nn.Sigmoid()
        self.prototype_features_MLP = MLP(2,latent_dim1,latent_dim1,self.num_graph_filter)
        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.layer_num+1):
            self.batch_norms.append(nn.BatchNorm1d(latent_dim1))
        
        self.node_encoder_list = nn.ModuleList()
        for _ in range(self.blocks_num):
            self.node_encoder_list.append(MLP(2, input_dim, latent_dim2*2,latent_dim1))
        


    def forward(self, W, features, targets, graph_indicator):
        unique, counts = torch.unique(graph_indicator, return_counts=True)
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
                # H = self.batch_norms[p*self.blocks_num + b](H)
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

        E_lists = []

        S_all = self.S_mlp(H_all)
        S_all = self.softmax(S_all)

        Cluster_assign_idx = [[] for i in range(len(unique))]
        count = 0
        
        for i in range(len(unique)):
            ind = (graph_indicator==i).nonzero(as_tuple=True)[0]
            H = H_all[ind]
            S = S_all[ind]
            S_ = torch.argmax(S, dim=1)
            for j in range(self.K_cluster):
                idx = (S_==j).nonzero(as_tuple=True)[0]
                Cluster_assign_idx[i].append(idx) ##
            E = S.t() @ H
            E_lists.append(E)
            count += len(ind)
        E_all = torch.stack(E_lists)

        E_list = E_all.reshape(E_all.shape[0],-1)
        
        pred1 = self.final(E_list)
        pred1 = F.dropout(pred1,self.dropout_rate,training = self.training)
        
           
        pred_softmax = torch.softmax(pred1, dim=-1)
        with torch.no_grad():
            sample_cat = torch.multinomial(pred_softmax, 1, replacement=False).flatten().to(device)
            ind_positive_sample = sample_cat == targets  # mark wrong sample results
            sample_cat_oh = F.one_hot(sample_cat, num_classes=pred1.shape[1]).float().to(device)
            epsilon = torch.where(sample_cat_oh != 0, 1 - pred_softmax, -pred_softmax).detach()
        sample_cat_oh = pred_softmax + epsilon


        Q = self.prototype_features_MLP(E_all)
        Q = F.softmax(Q,dim=-1)
        
        correlation_inter = F.softmax(self.correlation_inter, dim=0)
        correlation_intra = F.softmax(self.correlation_intra, dim=1)
        correlation = torch.mul(correlation_inter,correlation_intra)
        
        correlation_mask = sample_cat_oh @ correlation
        correlation_mask = correlation_mask.unsqueeze(1).repeat(1,self.K_cluster,1) #(128,2,4)

        prototype_mask = torch.mul(Q,correlation_mask)#(128,2,4)
        feature_mask = torch.sum(prototype_mask,dim=-1)
        E_sorted_list_mask = torch.einsum('bi,bij->bij', feature_mask, E_all)
        E_sorted_list_mask1 = E_sorted_list_mask.reshape(E_sorted_list_mask.shape[0],-1)

        pred2 = self.final(E_sorted_list_mask1)
        pred2 = F.dropout(pred2,self.dropout_rate,training = self.training)
        


        # out = {"pred1":pred1,"pred2":None,"S":S_all,"H":H_all,"ind_positive_sample":None,"E_all":E_all,"Q":None}
        out = {"pred1":pred1,"pred2":pred2,"S":S_all,"H":H_all,"ind_positive_sample":ind_positive_sample,"E_all":E_all,"Q":Q}
    
        return out
  

