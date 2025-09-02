import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Step_Loss(nn.Module):
    def __init__(self, args):
        super(Step_Loss, self).__init__()
        self.args = args
        self.cos_sim = nn.CosineSimilarity(dim=0)

    
    def forward(self,output,y,L,graph_indicator):

        ind_positive_sample = output['ind_positive_sample']
        S, H, Q = output['S'], output['H'], output['Q']
        E_all, L = output['E_all'], L
        batch, clf_labels = graph_indicator, y
        pred1, pred2 = output['pred1'], output['pred2']
        length = batch[-1]+1

        loss_1 = self.cross_entropy_loss(pred1, clf_labels)
        loss_2 = self.args.lambda_2*self.classSpecific_loss(pred2, clf_labels, ind_positive_sample)
        con_loss = self.ClusterConnectivity_loss(S, L, batch)
        con_loss = self.args.lambda_con*con_loss/length
        fea_loss = self.ClusterFeature_loss(S, H, batch)
        fea_loss = self.args.lambda_fea*fea_loss/length      
        proto_loss = self.args.lambda_proto*self.ClusterPrototype_loss(Q, E_all, S, batch)
    

        loss = loss_1 + loss_2 + con_loss + fea_loss + proto_loss

        return loss
    
    def cross_entropy_loss(self,pred1,clf_labels):
        loss = F.cross_entropy(pred1, clf_labels)
        return loss

    def classSpecific_loss(self, pred2,targets,ind_positive_sample):
        
        with torch.no_grad():
            n_positive_sample = int(torch.sum(ind_positive_sample))
        
        # Only use the correctly predicted samples to update the parameters
        if n_positive_sample != 0:
            loss_interpretation = F.cross_entropy(pred2[ind_positive_sample], targets.long().squeeze()[ind_positive_sample])
        else:
            loss_interpretation = torch.tensor(0.0).cuda()
        return loss_interpretation
    
    def ClusterPrototype_loss(self,Q,E,S_all,graph_indicator): 
        
        # Cluster prototype loss
        
        # Enforce that each column of matrix contains at least one value close to 1
        Q_expand = Q.reshape(-1,Q.shape[-1])
        similarity1 = 1-Q_expand
        loss1 = torch.mean(torch.min(similarity1, dim=0)[0])

        # Compute prototype embeddings
        n_proto = Q.shape[-1]
        Q_t = Q.permute(0,2,1)

        S_list = []
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        for i in range(len(unique)):
            ind = (graph_indicator==i).nonzero(as_tuple=True)[0]
            S = S_all[ind]
            count_S = torch.sum(S,dim=0)+1e-5
            S_list.append(count_S)
        
        S_list = torch.stack(S_list)
        count_S = torch.sum(S_list, dim=0)
        E = E/(count_S.unsqueeze(-1))
        prototype_embedding = torch.einsum('bmk,bkd->bmd', Q_t, E) #[128, 5, 32]
        count_cluster = torch.sum(Q,dim=1)+1e-5
        prototype_embedding = prototype_embedding/(count_cluster.unsqueeze(-1))
        prototype_embedding = torch.mean(prototype_embedding,0) #[5, 32]
        
        # Push apart different prototype embeddings
        center_loss = 0
        for j1 in range(n_proto):
            for j2 in range(n_proto):
                if j2 != j1:
                    mean1 = prototype_embedding[j1]
                    mean2 = prototype_embedding[j2]
                    center_loss += -torch.mean((mean1 - mean2)**2)
        center_loss = center_loss/((n_proto-1)*n_proto)
        

        loss = loss1+center_loss
        return loss

    def ClusterConnectivity_loss(self,S_softmax,L_sym,graph_indicator):
        # Cluster connectivity loss
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        k = S_softmax.size(-1)
        loss_sp = 0
        loss_ortho = 0
        i_s = torch.eye(k).type_as(S_softmax)
        L_sym = L_sym.to_dense()
        
        for i in range(len(unique)):
            ind = (graph_indicator==i).nonzero(as_tuple=True)[0]
            
            # Spectral loss
            S_i = S_softmax[ind]
            col_norm = torch.norm(S_i,p=2,dim=0)
            S_normalized_i = S_i / (col_norm + 1e-15)
            L_sym_i = L_sym[ind][:,ind]
            loss_sp_i = torch.trace(torch.matmul(torch.matmul(S_normalized_i.t(), L_sym_i), S_normalized_i))
            loss_sp += loss_sp_i
            ss = torch.matmul(S_normalized_i.transpose(0, 1), S_normalized_i)
           
            # Orthogonal loss
            loss_ortho_i = torch.norm(ss - i_s,p=2)
            loss_ortho += loss_ortho_i

        return loss_sp + loss_ortho

    def ClusterFeature_loss(self,S, H, graph_indicator):
   
        # Cluster feature loss
        n_clusters = S.shape[1]
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.size(0)
     
        # Reparameterization trick
        with torch.no_grad():
            sample_cat = torch.multinomial(S, 1, replacement=False).flatten().cuda()
            sample_cat_oh = F.one_hot(sample_cat, num_classes=S.shape[1]).float().cuda()
            epsilon = torch.where(sample_cat_oh != 0, 1 - S, -S).detach()
        S_mask = S + epsilon
        feature_loss = 0
        center_loss = 0
        for i in range(len(unique)):
            ind = (graph_indicator==i).nonzero(as_tuple=True)[0]
            H_i = H[ind]
            S_i = S_mask[ind]
            clusters = S_i.shape[1]
            feature_loss_i = 0
            center_loss_i = 0
            mean_list = []
            
            # Pull each node close to its cluster centroid
            for j in range(clusters):
                cluster_ind = torch.where(S_i[:,j] == 1)[0]
                mean_tensor = torch.sum(S_i[:,j][...,None] *H_i,0)
                len_cluster_ind = cluster_ind.shape[0]
                if len_cluster_ind > 0:
                    mean_tensor = mean_tensor/len_cluster_ind
                mean_list.append(mean_tensor)
                feature_distance = torch.mean(torch.sum((H_i[cluster_ind] - mean_tensor)**2, dim=0))
                if len_cluster_ind > 0:
                    feature_loss_i += feature_distance/len_cluster_ind

            
            feature_loss += feature_loss_i
           
            # Push apart different cluster centroids
            for j1 in range(clusters):
                for j2 in range(clusters):
                    if j2 != j1:
                        mean1 = mean_list[j1]
                        mean2 = mean_list[j2]
                        center_loss_i += torch.mean((mean1 - mean2)**2)
            center_loss += -center_loss_i
            center_loss = center_loss/(n_clusters-1)
        
        return (feature_loss + center_loss )/n_clusters
    

   



