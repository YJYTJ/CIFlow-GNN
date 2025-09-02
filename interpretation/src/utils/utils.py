import torch
import random
import numpy as np
import copy
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def set_seed(seed):
    if not seed:
        seed = 1

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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



def calculate_auc(filepath, epochs):
    max_fold = 1
    validation_loss = np.zeros((epochs, max_fold))
    test_accuracy = np.zeros((epochs, max_fold))
    test_acc = np.zeros(max_fold)
    with open(filepath, 'r') as filehandle:
        filecontents = filehandle.readlines()
        index = 0
        col = 0
        for line in filecontents:
            ss = line.split()
            t_acc = ss[1]
            v_loss = ss[0]
            validation_loss[index][col] = float(v_loss)
            test_accuracy[index][col] = float(t_acc)
            index += 1
            if index == epochs:
                index = 0
                col += 1
                if col == max_fold:
                    break

    min_ind = np.argmin(validation_loss, axis=0)
    for i in range(max_fold):
        ind = min_ind[i]
        test_acc[i] = test_accuracy[ind][i]
    ave_acc = np.mean(test_acc)
    std_acc = np.std(test_acc)
    return ave_acc