import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from src.datasets import  Mutag, Motif,SolubilityDataset,BenzeneDataset


def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, mutag_x=False):
    assert dataset_name in ['labeled-motifs','benzene','solubility', 'mutagenicity']

    if dataset_name == 'benzene':
        dataset = BenzeneDataset(data_dir/'benzene')
        split_idx = get_random_split_idx(dataset, splits)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    if dataset_name == 'solubility':
        dataset = SolubilityDataset(data_dir/'Solubility' )
        split_idx = get_random_split_idx(dataset, splits)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    elif dataset_name == 'mutagenicity':
        dataset = Mutag(root=data_dir / 'mutagenicity')
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]
    
    elif dataset_name == 'labeled-motifs' :
        train_set = Motif(root=data_dir / dataset_name, mode='train')
        valid_set = Motif(root=data_dir / dataset_name, mode='val')
        test_set = Motif(root=data_dir / dataset_name, mode='test')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': valid_set, 'test': test_set})

    x_dim = test_set[0].x.shape[1]
    if isinstance(test_set, list):
        num_class = Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]

    return loaders, test_set, x_dim, num_class


def get_random_split_idx(dataset, splits, random_state=None, mutag_x=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
    
        
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set
