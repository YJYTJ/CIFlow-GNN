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
        S,H,Q = output['S'],output['H'],output['Q']
        E_all,L = output['E_all'],L
        batch,clf_labels = graph_indicator,y
        pred1, pred2 = output['pred1'],output['pred2']
        length = batch[-1]+1

        loss_1 = self.cross_entropy_loss(pred1, clf_labels)
        loss_2 = self.args.lambda_2*self.classSpecific_loss(pred2,clf_labels,ind_positive_sample)
        con_loss = self.ClusterConnectivity_loss(S,L,batch)
        con_loss = self.args.lambda_con*con_loss/length
        fea_loss = self.ClusterFeature_loss(S, H,batch )
        fea_loss = self.args.lambda_fea*fea_loss/length      
        proto_loss = self.args.lambda_proto*self.ClusterPrototype_loss(Q,E_all)
    

        loss = loss_1 + loss_2 + con_loss + fea_loss + proto_loss

        return loss
    
    def ClusterPrototype_loss(self,Q,E): #每个prototype至少要与一个cluster相近
        # loss = torch.mean(torch.min(min_distances, dim=0)[0])
        Q_expand = Q.reshape(-1,Q.shape[-1])
        similarity1 = 1-Q_expand
        loss1 = torch.mean(torch.min(similarity1, dim=0)[0])
        ##每个cluster仅与一个prototype相近
        # similarity2 = 1-Q_expand
        # loss2 = torch.mean(torch.min(similarity2, dim=0)[0])
        
        #被分到同一类的protype彼此拉远
        clusters = Q.shape[-1]
        center_loss = 0

        Q_t = Q.permute(0,2,1)
        prototype_embedding = torch.einsum('bmk,bkd->bmd', Q_t, E)
        prototype_embedding = torch.sum(prototype_embedding,0)
        count_cluster = torch.sum(Q,dim=(0,1))+0.1
        prototype_embedding = prototype_embedding/(count_cluster.unsqueeze(1))
        prototype_embedding = prototype_embedding/(torch.norm(prototype_embedding,p=2,dim=1)+ 1e-15).unsqueeze(1)
        center_loss = 0
        for j1 in range(clusters):
            for j2 in range(j1+1,clusters):
                mean1 = prototype_embedding[j1,:]
                mean2 = prototype_embedding[j2,:]
                center_loss += -torch.mean((mean1 - mean2)**2)
        
        center_loss = center_loss/(clusters*(clusters-1)/2)

        loss = loss1+center_loss
        return loss
    
    def cross_entropy_loss(self,pred1,clf_labels):
        loss = F.cross_entropy(pred1, clf_labels)
        return loss

    def classSpecific_loss(self, pred2,targets,ind_positive_sample):
        
        with torch.no_grad():
            n_positive_sample = int(torch.sum(ind_positive_sample))
  
        if n_positive_sample != 0:
            # loss_interpretation2 = F.cross_entropy(pred2, targets)
            loss_interpretation3 = F.cross_entropy(pred2[ind_positive_sample], targets[ind_positive_sample])
            # loss_interpretation3 = F.cross_entropy(pred2, targets)
            loss_interpretation =  loss_interpretation3
        else:
            loss_interpretation = torch.tensor(0.0).cuda()
        return loss_interpretation


    def ClusterConnectivity_loss(self,S_softmax,L_sym,graph_indicator):

        unique, counts = torch.unique(graph_indicator, return_counts=True)
        k = S_softmax.size(-1)
        loss_sp = 0
        loss_ortho = 0
        i_s = torch.eye(k).type_as(S_softmax)*len(unique)
        S_normalized = torch.zeros_like(S_softmax).type_as(S_softmax)
        for i in range(len(unique)):
            ind = (graph_indicator==i).nonzero(as_tuple=True)[0]
            S_i = S_softmax[ind]
            col_norm = torch.norm(S_i,p=2,dim=0)
            S_normalized_i = S_i / (col_norm + 1e-05)
            S_normalized[ind] = S_normalized_i
        loss_sp = torch.trace(torch.matmul(S_normalized.t(),torch.sparse.mm(L_sym, S_normalized)))
        
        ss = torch.matmul(S_normalized.transpose(0, 1), S_normalized)
        loss_ortho = torch.norm(ss - i_s,p=2)
        


        return loss_sp + loss_ortho


    def ClusterFeature_loss(self,S, H, graph_indicator):

        n_clusters = S.shape[1]
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.size(0)
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
            
            for j1 in range(clusters):
                for j2 in range(j1+1,clusters):
                    mean1 = mean_list[j1]
                    mean2 = mean_list[j2]
                    center_loss_i += torch.mean((mean1 - mean2)**2)
            center_loss += -center_loss_i
            center_loss = center_loss/(n_clusters-1)/2
        
        return feature_loss + center_loss 
    

   



