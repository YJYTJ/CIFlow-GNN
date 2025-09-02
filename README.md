# CIFlow-GNN
![image text](https://github.com/YJYTJ/CIFlow-GNN/blob/main/flowchart.png "The pipeline of CIFlow-GNN")

🔥**CIFlow-GNN**: "Enhancing Graph Learning Interpretability through Cluster Information Flow", Jiayi Yang, Wei Ye*, Xin Sun, Rui Fan, and Jungong Han. This repository contains the official PyTorch implementation of our work.

## 🚀 About

**CIFlow-GNN** is an efficient subgraph-based built-in GNN explainer that provides explanations by modulating the cluster information flow. The framework is designed to enhance both graph interpretability and graph prediction performance.

## ⚡ Changelog

[25/09/02] Added all ten datasets with corresponding splits for graph classification.  
Included dataset descriptions for graph interpretation.  
Provided hyperparameter configurations for both graph interpretation and graph classification.


## 🎓 graph interpretation
### Requirements
To set up the environment for graph interpretation:
```shell
cd interpretation
pip install -r requirements.txt
```

### Datasets
The datasets for graph interpretation are available in the ```./interpretation/data``` folder, including:
- **Labeled-Motifs**
- **Mutagenicity**
- **Solubility**
- **Benzene**

To generate the new synthetic dataset (**Labeled-Motifs**) proposed by us, run the following:
```shell
cd interpretation
python data/labeled-motifs/generate_dataset/motifgen.py
```

The details of four datasets are provided in ```./interpretation/dataset_description.md```.

### Options
- `--lambda_2`: Hyper-parameter for interpretation $\mathcal{L}_{\text{2}}$ loss.
- `--lambda_con`: Hyper-parameter for cluster connectivity loss.
- `--lambda_fea`: Hyper-parameter for cluster feature loss.
- `--lambda_proto`: Hyper-parameter for cluster prototype loss.
- `--layer_num`: Number of layers for gnn.
- `--clusters`: Number of predefined clusters for spectral graph clustering.
- `--num_graph_filter`: Number of predefined graph filters for all classes in the dataset.
- `--important_c`: Select the top-c clusters as important subgraphs.
- `--result_folder`: Path to the output folder.
- `--dataset_name`: Datasets for graph interpretation (solubility/benzene/mutagenicity/labeled-motifs).

### Reproduce Results
We provide the source code to reproduce the results in our paper. To reproduce Table 3 (ROC AUC) in CIFlow-GNN, one needs to run the code in `interpretation/run.sh` as follows:

```shell
cd interpretation

# labeled-motifs
python main.py --dataset_name labeled-motifs --lambda_2 0.3 --lambda_con 0.05 --lambda_fea 0.1 --lambda_proto 0.15 --layer_num 4 --clusters 2 --num_graph_filter 3 --epochs 100

# mutagenicity
python main.py --dataset_name mutagenicity --lambda_2 0.3 --lambda_con 0.01 --lambda_fea 0.1 --lambda_proto 0.05 --layer_num 4 --clusters 4 --num_graph_filter 6 --epochs 100

# solubility
python main.py --dataset_name solubility --lambda_2 1.0 --lambda_con 0.05 --lambda_fea 0.01 --lambda_proto 0.05 --layer_num 5 --clusters 4 --num_graph_filter 5 --epochs 100

# benzene
python main.py --dataset_name benzene --lambda_2 0.5 --lambda_con 0.1 --lambda_fea 0.3 --lambda_proto 0.1 --layer_num 4 --clusters 3 --num_graph_filter 5 --epochs 100
```


## 🎓 graph classification
### Requirements
To set up the environment for graph classification:
```shell
cd classification
pip install -r requirements.txt
```

### Datasets
-  We provide all ten datasets in the ```./classification/datasets.zip```. Unzip the ```./classification/datasets.zip``` and place it in the ```./classification/datasets``` folder. The datasets used for the graph classification task are MUTAG, BZR, BZR_MD, DHFR, COX2, PROTEINS, NCI1, DD, IMDB-BINARY, and IMDB-MULTI.
-  For fair comparison, we use the cross-validation splits strategy provided by [this repository](https://github.com/diningphil/gnn-comparison). Data splits for all ten datasets are provided in ```./classification/data_split_dir.zip```. Unzip the ```./classification/data_split_dir.zip``` and place it in the ```./classification/data_split_dir``` folder.


### Reproduce Results
We provide the source code to reproduce the results in our paper. To reproduce Table 7 (Accuracy) in CIFlow-GNN, one needs to run the code in `classification/run.sh` as follows:

```shell
cd classification

# MUTAG
python main.py --dataset_name MUTAG --latent_dim1 32 --latent_dim2 256 --lambda_2 0.5 --lambda_con 0.01 --lambda_fea 0.05 --lambda_proto 0.05 --layer_num 5 --clusters 2 --num_graph_filter 4 --epochs 350 --use_node_labels

# BZR
python main.py --dataset_name BZR --latent_dim1 32 --latent_dim2 256 --lambda_2 0.3 --lambda_con 0.01 --lambda_fea 0.01 --lambda_proto 0.05 --layer_num 3 --clusters 3 --num_graph_filter 3 --epochs 350 --use_node_labels

# BZR_MD
python main.py --dataset_name BZR_MD --latent_dim1 32 --latent_dim2 32 --lambda_2 0.5 --lambda_con 0.01 --lambda_fea 0.05 --lambda_proto 0.05 --layer_num 5 --clusters 2 --num_graph_filter 3 --epochs 350 --use_node_labels

# DHFR
python main.py --dataset_name DHFR --latent_dim1 32 --latent_dim2 128 --lambda_2 0.1 --lambda_con 0.01 --lambda_fea 0.01 --lambda_proto 0.05 --layer_num 3 --clusters 2 --num_graph_filter 3 --epochs 350 --use_node_labels

# COX2
python main.py --dataset_name COX2 --latent_dim1 32 --latent_dim2 128 --lambda_2 0.3 --lambda_con 0.01 --lambda_fea 0.01 --lambda_proto 0.05 --layer_num 4 --clusters 3 --num_graph_filter 3 --epochs 350 --use_node_labels

# PROTEINS
python main.py --dataset_name PROTEINS --latent_dim1 32 --latent_dim2 32 --lambda_2 0.1 --lambda_con 0.01 --lambda_fea 0.01 --lambda_proto 0.05 --layer_num 3 --clusters 2 --num_graph_filter 3 --epochs 350 --use_node_labels

# NCI1
python main.py --dataset_name NCI1 --latent_dim1 32 --latent_dim2 32 --lambda_2 0.1 --lambda_con 0.01 --lambda_fea 0.01 --lambda_proto 0.05 --layer_num 3 --clusters 2 --num_graph_filter 4 --epochs 350 --use_node_labels

# DD
python main.py --dataset_name DD --latent_dim1 32 --latent_dim2 32 --lambda_2 0.3 --lambda_con 0.05 --lambda_fea 0.01 --lambda_proto 0.05 --layer_num 6 --clusters 3 --num_graph_filter 4 --epochs 350 --use_node_labels

# IMDB-BINARY
python main.py --dataset_name IMDB-BINARY --latent_dim1 32 --latent_dim2 32 --lambda_2 0.1 --lambda_con 0.01 --lambda_fea 0.01 --lambda_proto 0.05 --layer_num 4 --clusters 3 --num_graph_filter 5 --epochs 350

# IMDB-MULTI
python main.py --dataset_name IMDB-MULTI --latent_dim1 32 --latent_dim2 64 --lambda_2 0.1 --lambda_con 0.01 --lambda_fea 0.01 --lambda_proto 0.05 --layer_num 4 --clusters 2 --num_graph_filter 4 --epochs 350
```


## 🗨️ Contacts

For more details about our article, feel free to reach out to us. We are here to provide support and answer any questions you may have. 

- **Email**: Send us your inquiries at [2111125@tongji.edu.cn].


