from BA3_loc import *
import pickle
from tqdm import tqdm
import os.path as osp
import warnings
warnings.filterwarnings("ignore")
import random


data_dir = 'data/labeled-motifs/raw/'

if not osp.exists(data_dir):
    os.makedirs(data_dir)


def get_Tree(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):

    list_shapes = [["Tree_motif"]] * nb_shapes # house

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph1(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

def get_Hexagon(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:

    Start with a tree and attach cycle-shaped (directed edges) subgraphs.
    """
    list_shapes = [["Hexagon_motif"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

def get_Grid(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:

    Start with a tree and attach crane-shaped subgraphs.
    """
    list_shapes = [["Grid_motif"]] * nb_shapes   # crane

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.00, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


edge_index_list, label_list = [], []
ground_truth_list, role_id_list, pos_list = [], [], []


graph_num=300

def graph_stats():
    base = 'Tree_motif'
    width_basis=np.random.choice(range(2,3))
    return base, width_basis

e_mean, n_mean = [], []
for _ in tqdm(range(graph_num)): # Tree+Hexagon
    base, width_basis = graph_stats()

    G, role_id, name = get_Hexagon(basis_type=base, nb_shapes=1, 
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

e_mean, n_mean = [], []
for _ in tqdm(range(graph_num)): # Tree+Grid
    base, width_basis = graph_stats()
    G, role_id, name = get_Grid(basis_type=base, nb_shapes=1, 
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #sEdges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

e_mean, n_mean = [], []
for _ in tqdm(range(graph_num*2)): #Tree
    base, width_basis = graph_stats()
    G, role_id, name = get_Tree(basis_type=base, nb_shapes=1, 
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #sEdges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))


with open(osp.join(data_dir, 'train.npy'), 'wb') as f:
    pickle.dump((edge_index_list, label_list, ground_truth_list, role_id_list, pos_list), f, protocol=pickle.HIGHEST_PROTOCOL)


edge_index_list, label_list = [], []
ground_truth_list, role_id_list, pos_list = [], [], []
# bias = float(global_b)

e_mean, n_mean = [], []
for _ in tqdm(range(100)):
    base, width_basis = graph_stats()
    G, role_id, name = get_Hexagon(basis_type=base, nb_shapes=1, 
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

e_mean, n_mean = [], []
for _ in tqdm(range(100)):
    base, width_basis = graph_stats()

    G, role_id, name = get_Grid(basis_type=base, nb_shapes=1, 
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #sEdges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

e_mean, n_mean = [], []
for _ in tqdm(range(200)):
    base, width_basis = graph_stats()

    G, role_id, name = get_Tree(basis_type=base, nb_shapes=1, 
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #sEdges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

zipped_lists = list(zip(edge_index_list, label_list,ground_truth_list,role_id_list,pos_list))
random.shuffle(zipped_lists)
edge_index_list, label_list,ground_truth_list,role_id_list,pos_list = zip(*zipped_lists)
with open(osp.join(data_dir, 'val.npy'), 'wb') as f:
    pickle.dump((edge_index_list, label_list, ground_truth_list, role_id_list, pos_list), f, protocol=pickle.HIGHEST_PROTOCOL)



edge_index_list, label_list = [], []
ground_truth_list, role_id_list, pos_list = [], [], []

e_mean, n_mean = [], []
for _ in tqdm(range(100)):
    base, width_basis = graph_stats()

    G, role_id, name = get_Hexagon(basis_type=base, nb_shapes=1, 
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

e_mean, n_mean = [], []
for _ in tqdm(range(100)):
    base, width_basis = graph_stats()

    G, role_id, name = get_Grid(basis_type=base, nb_shapes=1, 
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #sEdges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

e_mean, n_mean = [], []
for _ in tqdm(range(200)):
    base, width_basis = graph_stats()

    G, role_id, name = get_Tree(basis_type=base, nb_shapes=1, 
                                    width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #sEdges: %.2f " % (len(ground_truth_list), np.mean(n_mean), np.mean(e_mean)))

zipped_lists = list(zip(edge_index_list, label_list,ground_truth_list,role_id_list,pos_list))
random.shuffle(zipped_lists)
edge_index_list, label_list,ground_truth_list,role_id_list,pos_list = zip(*zipped_lists)

with open(osp.join(data_dir, 'test.npy'), 'wb') as f:
    pickle.dump((edge_index_list, label_list, ground_truth_list, role_id_list, pos_list), f, protocol=pickle.HIGHEST_PROTOCOL)

