import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torch_geometric.utils import to_dense_adj

def draw_graph(ResultFolderPATH3,values,epoch,Cluster_assign_idx,test_set,best_cluster,class_c,prototype_m):
    
    # predefine_nodes_color={0:'#F1BD4D',1:'#85ADCF',2:'#C0D6E6',3:'#29477E'}
    predefine_nodes_color={0:'#880015',1:'#ED1C24',2:'#FF7F27',3:'#FFF200',4:'#22B14C',5:'#00A2E8',6:'#3F48CC',7:'#A349A4',8:'#880015',9:'#ED1C24',10:'#FF7F27',11:'#FFF200',12:'#22B14C',13:'#00A2E8',14:'#3F48CC',15:'#A349A4'}
    count = 0
    for key,v in best_cluster.items():
        [graph_n,k] = v
        graph = test_set[graph_n]
        adj_matrix = to_dense_adj(graph.edge_index,max_num_nodes=graph.num_nodes)
        adj = np.array(adj_matrix.cpu())[0]
        node_label = np.argmax(np.array(graph.x.cpu()),1)
        num_nodes = graph.num_nodes
        # initial_node_labels = {id_val: label_val for id_val, label_val in zip(ind, node_label)}
        G = nx.Graph()

        # 添加节点和边
        G.add_nodes_from(range(num_nodes))
        for kk in range(num_nodes):
            for j in range(kk + 1, num_nodes):
                if adj[kk][j] == 1:
                    G.add_edge(kk, j)

        important_nodes = Cluster_assign_idx[graph_n][k]
        
        # 计算每条边的宽度
        edge_widths = []
        for edge in G.edges():
            if edge[0] in important_nodes and edge[1] in important_nodes:
                edge_widths.append(5)  # 加粗的边
            else:
                edge_widths.append(1)  # 普通的边

        # 绘制图形
        plt.clf()
        pos = nx.spring_layout(G)
        draw_node_colors = [predefine_nodes_color[node_label[node]] for node in G.nodes()]
        draw_node_labels = {node: node_label[node] for node in G.nodes()}
        nx.draw(G, pos, with_labels=True, labels=draw_node_labels,width=edge_widths, node_size=50, node_color=draw_node_colors, cmap=plt.cm.tab10)
        plt.title('Graph{}'.format(count))
        count = count + 1
        # plt.show()

        ResultFolderPATH = ResultFolderPATH3+'/img/epoch{}/class{}_prototype{}_{}/'.format(epoch,class_c,prototype_m,values)
        if not os.path.exists(ResultFolderPATH):
            os.makedirs(ResultFolderPATH)
        plt.savefig(ResultFolderPATH+'graph{}.png'.format(key))



    
    
    
    
    # adj_matrix = np.array(adj_matrix.cpu())
    # node_labels = np.array(node_labels.cpu())
    # graph_indicator = np.array(graph_indicator.cpu())
    # unique, counts = np.unique(graph_indicator, return_counts=True)
    # # S = S.cpu().detach().numpy() ########.cpu()
    # feature_mask = feature_mask.cpu().detach().numpy()
    # max_cluster = np.argmax(feature_mask,1)
    # predefine_nodes_color={0:'#F1BD4D',1:'#85ADCF',2:'#C0D6E6',3:'#29477E'}
    # #0:黄，
    # for i in range(len(unique)):
    #     ind = np.nonzero(graph_indicator == i)[0]
    #     important_nodes = Cluster_assign_idx[i][max_cluster[i]]

    #     # s = np.argmax(S[ind],1)
    #     adj = adj_matrix[ind][:,ind]
    #     node_label = np.argmax(node_labels[ind], 1)
    #     ind = ind - ind.min()
    #     initial_node_labels = {id_val: label_val for id_val, label_val in zip(ind, node_label)}
    #     # node_classes = {id_val: cluster_val for id_val, cluster_val in zip(ind, s)}

    #     G = nx.Graph()

    #     # 添加节点和边
    #     num_nodes = len(ind)
    #     G.add_nodes_from(range(num_nodes))
    #     for k in range(num_nodes):
    #         for j in range(k + 1, num_nodes):
    #             if adj[k][j] == 1:
    #                 G.add_edge(k, j)
        
    #     # 计算每条边的宽度
    #     edge_widths = []
    #     for edge in G.edges():
    #         if edge[0] in important_nodes and edge[1] in important_nodes:
    #             edge_widths.append(5)  # 加粗的边
    #         else:
    #             edge_widths.append(1)  # 普通的边

    #     # 绘制图形
    #     plt.clf()
    #     pos = nx.spring_layout(G)
    #     draw_node_colors = [predefine_nodes_color[initial_node_labels[node]] for node in G.nodes()]
    #     draw_node_labels = {node: initial_node_labels[node] for node in G.nodes()}
    #     nx.draw(G, pos, with_labels=True, labels=draw_node_labels,width=edge_widths, node_size=500, node_color=draw_node_colors, cmap=plt.cm.tab10)
    #     plt.title('Graph{}'.format(count+i))
    #     # plt.show()

    #     ResultFolderPATH = '/home/jiayi/PyTorch/GSAT-main/gsat_proto_6_20/img/'
    #     if not os.path.exists(ResultFolderPATH):
    #         os.makedirs(ResultFolderPATH)
    #     plt.savefig(ResultFolderPATH+'graph{}.png'.format(count+i))
    # return count+len(unique)



# 1 feature + spe+orth
#2 feautre
