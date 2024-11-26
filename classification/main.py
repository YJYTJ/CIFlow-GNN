import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import json
import time
import argparse
import numpy as np
from math import ceil
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.preprocessing import LabelEncoder
from model import CIFlow
from utils import *
from calculate_average_accuracy import *
import random
from losses import *
def seed_all(seed):
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

def args_parser():

    parser = argparse.ArgumentParser(description='CIFlow-GNN')

    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train')
    parser.add_argument('--dataset_name', default='MUTAG', help='dataset name')
    parser.add_argument('--use_node_labels', action='store_true', default=False, help='whether to use node labels')
    parser.add_argument('--use_node_attri', action='store_true', default=False, help='whether to use node attributes')
    parser.add_argument('--latent_dim1', type=int, default=32)
    parser.add_argument('--latent_dim2', type=int, default=64, help='node_feature hidden')
    parser.add_argument('--final_mlp_dim', type=int, default=16, help='final hidden')
    parser.add_argument('--max_step', type=int, default=1, help='max length of random walks')
    parser.add_argument('--dropout_rate', type=float, default=0.6, help='dropout rate (1 - keep probability).')
    parser.add_argument('--no_norm', action='store_true', default=False, help='whether to apply normalization')
    parser.add_argument('--max_fold', type=int, default=11, help='max_fold')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--lambda_2', type=float, default=0.1)
    parser.add_argument('--lambda_con', type=float, default=0.1)
    parser.add_argument('--lambda_fea', type=float, default=0.01)
    parser.add_argument('--lambda_proto', type=float, default=0.15)
    parser.add_argument('--K_cluster', type=int, default=3, help='K_cluster')
    parser.add_argument('--num_graph_filter', type=int, default=5, help='num_graph_filter')
    parser.add_argument('--result_folder',  type=str, default='results', help='results folder')
    parser.add_argument('--power_num',  type=int, default=6, help='W power_num')
    parser.add_argument('--blocks_num',  type=int, default=1, help='block number') 
    args = parser.parse_args()
    return args


def main(): 
    args = args_parser()
    seed_all(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    adj_lst, features_lst, class_labels = load_data(args.dataset_name, args.use_node_labels,use_node_attri=args.use_node_attri)
    N = len(adj_lst)
    features_dim = features_lst[0].shape[1]
    enc = LabelEncoder()
    class_labels = enc.fit_transform(class_labels)
    n_classes = np.unique(class_labels).size
    y = [np.array(class_labels[i]) for i in range(class_labels.size)]

  
    for num_fold in range(1,args.max_fold):
        directory = "data_split_dir/{0}/10fold_idx".format(args.dataset_name)
        train_file = "train_idx-{0}.txt".format(num_fold)
        train_index = []
        with open(os.path.join(directory, train_file), 'r') as file:
            for line in file:
                train_index.append(int(line.rstrip()))

        val_file = "validation_idx-{0}.txt".format(num_fold)
        val_index = []
        with open(os.path.join(directory, val_file), 'r') as file:
            for line in file:
                val_index.append(int(line.rstrip()))

        test_file = "test_idx-{0}.txt".format(num_fold)
        test_index = []
        with open(os.path.join(directory, test_file), 'r') as file:
            for line in file:
                test_index.append(int(line.rstrip()))


        n_test = len(test_index)
        n_train = len(train_index)
        n_val = len(val_index)

        # Sampling
        adj_train = [adj_lst[i] for i in train_index]
        features_train = [features_lst[i] for i in train_index]
        y_train = [y[i] for i in train_index]

        adj_test = [adj_lst[i] for i in test_index]
        features_test = [features_lst[i] for i in test_index]
        y_test = [y[i] for i in test_index]

        adj_val = [adj_lst[i] for i in val_index]
        features_val = [features_lst[i] for i in val_index]
        y_val = [y[i] for i in val_index]

        # Create batches
        adj_test, features_test, graph_indicator_test, y_test = generate_batches(adj_test, features_test, y_test,args.batch_size, device)
        adj_train, features_train, graph_indicator_train, y_train = generate_batches(adj_train, features_train, y_train,args.batch_size, device)
        adj_val, features_val, graph_indicator_val, y_val = generate_batches(adj_val, features_val, y_val, args.batch_size, device)
        Lsym_train,W_train = normalized_laplacian_W_batch(adj_train,args.power_num)
        Lsym_val,W_val = normalized_laplacian_W_batch(adj_val,args.power_num)
        Lsym_test,W_test = normalized_laplacian_W_batch(adj_test,args.power_num)
    

        n_test_batches = ceil(n_test/args.batch_size)
        n_train_batches = ceil(n_train/args.batch_size)
        n_val_batches = ceil(n_val/args.batch_size)

        # Create model
        model = CIFlow(args, features_dim, n_classes).to(device)
        
        # set up training
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = Step_Loss(args)

        def train(adj,W, L,features, graph_indicator, y, batches, n):
            model.train()
            loss_all = 0
            correct = 0
            for i in range(batches):
                optimizer.zero_grad()  
                output = model(W[i],features[i],y[i], graph_indicator[i])
                pred1= output['pred1']
                loss_train = criterion.forward(output,y[i],L[i],graph_indicator[i])
                prediction1 = pred1.max(1)[1]
                correct += prediction1.eq(y[i].data).sum().item()
                loss_train.backward()
                loss_all += pred1.size(0) * loss_train.item() 
                optimizer.step()
                
                
            return correct/n, loss_all/n

        def val_test(adj,W, L, features, graph_indicator, y,batches,n):
            model.eval()
            loss_all = 0
            correct = 0
 
            for i in range(batches):
                output = model(W[i],features[i],y[i], graph_indicator[i])
                pred1= output['pred1']
                loss_val = criterion.forward(output,y[i],L[i],graph_indicator[i])
                prediction1 = pred1.max(1)[1]
                correct += prediction1.eq(y[i].data).sum().item()          
                loss_all +=  pred1.size(0) * loss_val.item()
                
            return correct/n, loss_all/n

        ResultFolderPATH = args.result_folder +'/'+args.dataset_name+ '/'  \
            + 'latent_dim1_'+ str(args.latent_dim1)+'/'+ 'latent_dim2_'+ str(args.latent_dim2)+ '/'  \
            + 'lambda_con' + str(args.lambda_con)+ '/'  \
            + 'lambda_fea' + str(args.lambda_fea)+ '/'  \
            + 'lambda_proto' + str(args.lambda_proto)+ '/'  \
            + 'lambda_2' + str(args.lambda_2)+ '/'  \
            + 'K_cluster' + str(args.K_cluster)+ '/'  \
            + 'power_num' + str(args.power_num)+ '/'  \
            + 'blocks_num' + str(args.blocks_num)+ '/'  \
            + 'num_graph_filter' + str(args.num_graph_filter)


        if not os.path.exists(ResultFolderPATH):
            os.makedirs(ResultFolderPATH)
        acc_per_epoch = np.zeros((args.epochs, 3))
        
        for epoch in range(args.epochs):
            train_acc, train_loss = train(adj_train,W_train, Lsym_train, features_train, graph_indicator_train, y_train,n_train_batches,n_train)
            val_acc, val_loss = val_test(adj_val,W_val, Lsym_val, features_val, graph_indicator_val, y_val,n_val_batches,n_val)
            test_acc, test_loss = val_test(adj_test, W_test ,Lsym_test, features_test, graph_indicator_test, y_test, n_test_batches,n_test)
            acc_per_epoch[epoch][0] = val_loss
            acc_per_epoch[epoch][1] = test_acc
            
            print(
                ' Epoch: {:02d}, trainloss: {:.4f}, train_acc: {:.4f}'
                .format(epoch, train_loss, train_acc))
            print(
                ' Epoch: {:02d}, valloss: {:.4f}, val_acc: {:.4f}'
                .format(epoch, val_loss, val_acc))
            print(
                ' Epoch: {:02d}, testloss: {:.4f}, test_acc: {:.4f}'
                .format(epoch, test_loss, test_acc))
        
    
        with open(ResultFolderPATH + '/acc_results.txt', 'a+') as f:
            for epoch in range(args.epochs):
                f.write(str(acc_per_epoch[epoch][0]) + ' ')
                f.write(str(acc_per_epoch[epoch][1]) + '\n')
              
    

    with open(ResultFolderPATH + '/acc_results.txt', 'a+') as f:
        f.write(str(calculate_acc(ResultFolderPATH + '/acc_results.txt',args.epochs,args.max_fold-1)))
    print(str(calculate_acc(ResultFolderPATH + '/acc_results.txt',args.epochs,args.max_fold-1)))
    


if __name__ == "__main__":
    main()