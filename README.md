# CIFlow-GNN
![image text](https://github.com/YJYTJ/CIFlow-GNN/blob/main/flowchart.jpg "The pipeline of CIFlow-GNN")

🔥**CIFlow-GNN**: "Enhancing Graph Learning Interpretability through Cluster Information Flow", Jiayi Yang, Wengang Guo, Xing Wei, Zexi Huang, and Wei Ye. This repository contains the official PyTorch implementation of our work.

## 🚀 About

**CIFlow-GNN** is an efficient subgraph-based built-in GNN explainer that provides explanations by modulating the cluster information flow. The framework is designed to enhance both graph interpretability and graph prediction performance.

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

### Options
- `--lambda_2`: Hyper-parameter for interpretation $\mathcal{L}_{\text{2}}$ loss.
- `--lambda_con`: Hyper-parameter for cluster connectivity loss.
- `--lambda_fea`: Hyper-parameter for cluster feature loss.
- `--lambda_proto`: Hyper-parameter for cluster prototype loss.
- `--clusters`: Number of predefined clusters for spectral graph clustering.
- `--num_graph_filter`: Number of predefined graph filters for all classes in the dataset.
- `--important_c`: Select the top-c clusters as important subgraphs.
- `--result_folder`: Path to the output folder.
- `--dataset_name`: Datasets for graph interpretation (solubility/benzene/Mutagenicity/labeled-motifs).

### Example
Run the graph interpretation task with the **Labeled-Motifs** dataset:
```shell
cd interpretation
python main.py --dataset_name labeled-motifs
```

## 🎓 graph classification
### Requirements
To set up the environment for graph classification:
```shell
cd classification
pip install -r requirements.txt
```

### Datasets
-  [Download](https://chrsmrrs.github.io/datasets/docs/datasets/) the datasets and place them in the ```./classification/datasets``` folder.
-  For fair comparison, use the cross-validation splits strategy provided by [this repository](https://github.com/diningphil/gnn-comparison). Copy the data splits to ```./classification/data_split_dir```.

To quickly get started, we include the **MUTAG** dataset along with its data splits in our repository.

### Example
```shell
cd classification
python main.py --dataset_name MUTAG
```

## 🗨️ Contacts

For more details about our article, feel free to reach out to us. We are here to provide support and answer any questions you may have. 

- **Email**: Send us your inquiries at [2111125@tongji.edu.cn].


