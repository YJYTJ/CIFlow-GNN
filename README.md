# CIFlow-GNN
![image text](https://github.com/YJYTJ/CIFlow-GNN/blob/main/flowchart.jpeg "The pipeline of CIFlow-GNN")
🔥
This repository is the official PyTorch implementation of "Enhancing Graph Learning Interpretability through Cluster Information Flow", Jiayi Yang, Wengang Guo, Xing Wei, Zexi Huang, and Wei Ye.

## 🚀 About

**CIFlow-GNN** is an efficient subgraph-based built-in graph explainer that can provide explanations for graphs and the graph model simultaneously by modulating the information flow. We evaluate CIFlow-GNN in terms of both interpretation and prediction performance.  



## 🎓 graph interpretation
### Requirements

```shell
cd interpretation
pip install -r requirements.txt
```

### Datasets
The repository contains the datasets for graph interpretation in the ./interpretation/data folder, including Labeled-Motifs, Mutagenicity, Solubility, and Benzene. Specifically, you can generate the new synthetic dataset Labeled-Motifs by taking the following steps.
```shell
cd interpretation/data/labeled-motifs/generate_dataset/motif_gen
python motifgen.py
```

### Options
- `--lambda_2`: hyper-parameter controlling the weights of interpretation $\mathcal{L}_{\text{2}}$ loss.
- `--lambda_con`: hyper-parameter controlling the weights of cluster connectivity loss.
- `--lambda_fea`: hyper-parameter controlling the weights of cluster feature loss.
- `--lambda_proto`: hyper-parameter controlling the weights of cluster prototype loss.
- `--clusters`: the predefined number of clusters for spectral graph clustering.
- `--num_graph_filter`: the predefined number of graph filters for all classes in the dataset.
- `--important_c`: select the top-c clusters as the important subgraphs.
- `--result_folder`: the output folder.
- `--dataset_name`: datasets for graph interpretation (solubility/benzene/Mutagenicity/labeled-motifs).

### Example
```shell
cd interpretation
python main.py --dataset_name labeled-motifs
```



## 🎓 graph classification
### Requirements

```shell
cd classification
pip install -r requirements.txt
```

### Datasets
All the 10 datasets used for graph classification task can be download [here](https://chrsmrrs.github.io/datasets/docs/datasets/) Please download the datasets to the ./classification/datasets folder. For a fair comparison, we followed the cross-validation procedure in this [repository](https://github.com/diningphil/gnn-comparison) to get data splits. Please copy the splits to the folder ./classification/data_split_dir. We provide the MUTAG dataset and its data splits. You can use it for a quick start！

### Example
```shell
cd classification
python main.py --dataset_name MUTAG
```

The repository contains the datasets for graph interpretation in the ./interpretation/data folder, including Labeled-Motifs, Mutagenicity, Solubility, and Benzene. Specifically, you can generate the new synthetic dataset Labeled-Motifs by taking the following steps.
```shell
cd interpretation/data/labeled-motifs/generate_dataset/motif_gen
python motifgen.py
```

provide the code for generating the new synthetic dataset Labeled-Motifs. W

a jupyter notebook with an example code that produces model-level explanations for the BA-house dataset.



## 🗨️ Contacts

For more details about our article, feel free to reach out to us. We are here to provide support and answer any questions you may have. 

- **Email**: Send us your inquiries at [2111125@tongji.edu.cn].


