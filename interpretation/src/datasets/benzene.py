from pathlib import Path
from typing import Tuple

import torch
import pandas as pd
from pandas.api.types import CategoricalDtype
import math
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
# import torchgraphs as tg


def extract_unique_atoms(smiles_list,dataset_num):
    atom_set = set()
    for item in range(dataset_num):
        smiles = smiles_list['smiles'].iloc[item]
        mol = Chem.MolFromSmiles(smiles)
        # mol_with_hydrogens = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            atom_set.add(atom.GetSymbol())
    # return sorted(atom_set)
    unique_atoms = sorted(atom_set)
    atom_label_dict = {atom: idx for idx, atom in enumerate(unique_atoms)}
    return atom_label_dict


    # return tg.Graph(
    #     num_nodes=molecule.GetNumAtoms(),
    #     num_edges=molecule.GetNumBonds() * 2,
    #     node_features=node_features,
    #     edge_features=edge_features,
    #     senders=senders,
    #     receivers=receivers
    # )


# def smiles_to_graph(smiles: str):
#     molecule = Chem.MolFromSmiles(smiles)

#     atoms_df = []
#     for i in range(molecule.GetNumAtoms()):
#         atom = molecule.GetAtomWithIdx(i)
#         atoms_df.append({
#             'index': i,
#             'symbol': atom.GetSymbol(),
#             'degree': atom.GetDegree(),
#             'hydrogens': atom.GetTotalNumHs(),
#             'impl_valence': atom.GetImplicitValence(),
#         })
#     atoms_df = pd.DataFrame.from_records(atoms_df, index='index',
#                                          columns=['index', 'symbol', 'degree', 'hydrogens', 'impl_valence'])
#     atoms_df.symbol = atoms_df.symbol.astype(symbols)

#     node_features = torch.tensor(pd.get_dummies(atoms_df, columns=['symbol']).values, dtype=torch.float)

#     bonds_df = []
#     for bond in molecule.GetBonds():
#         bonds_df.append({
#             'sender': bond.GetBeginAtomIdx(),
#             'receiver': bond.GetEndAtomIdx(),
#             'type': bond.GetBondType().name,
#             'conj': bond.GetIsConjugated(),
#             'ring': bond.IsInRing()
#         })
#         bonds_df.append({
#             'sender': bond.GetEndAtomIdx(),
#             'receiver': bond.GetBeginAtomIdx(),
#             'type': bond.GetBondType().name,
#             'conj': bond.GetIsConjugated(),
#             'ring': bond.IsInRing()
#         })
#     bonds_df = pd.DataFrame.from_records(bonds_df, columns=['sender', 'receiver', 'type', 'conj', 'ring'])\
#         .set_index(['sender', 'receiver'])
#     bonds_df.conj = bonds_df.conj * 2. - 1
#     bonds_df.ring = bonds_df.ring * 2. - 1
#     bonds_df.type = bonds_df.type.astype(bonds)

#     edge_features = torch.tensor(pd.get_dummies(bonds_df, columns=['type']).values.astype(float), dtype=torch.float)
#     senders = torch.tensor(bonds_df.index.get_level_values('sender'), dtype=torch.long)
#     receivers = torch.tensor(bonds_df.index.get_level_values('receiver'), dtype=torch.long)

#     return
    # return tg.Graph(
    #     num_nodes=molecule.GetNumAtoms(),
    #     num_edges=molecule.GetNumBonds() * 2,
    #     node_features=node_features,
    #     edge_features=edge_features,
    #     senders=senders,
    #     receivers=receivers
    # )


class BenzeneDataset(InMemoryDataset):
    def __init__(self, root):
        # self.file_path = path+'/delaney-processed.csv'
        super().__init__(root)
        self.root = root
        # self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        # self.df = pd.read_csv(path)
        # self.df['molecules'] = self.df.smiles.apply(smiles_to_graph)
    @property
    def processed_file_names(self):
        return ['benzene.pt']
    
    def process(self):
        self.df = pd.read_csv(str(self.root)+'/benzene_smiles.csv')
        dataset_len = len(self.df)
        unique_atoms_dict = extract_unique_atoms(self.df,dataset_len)
        unique_atoms_num = len(unique_atoms_dict)
        self.data_list = []
        categories = [list(range(unique_atoms_num))]
        encoder = OneHotEncoder(categories=categories)
        # benzene_smarts = 'c1ccccc1'
        # benzene = Chem.MolFromSmarts(benzene_smarts)
        
        

        for item in range(dataset_len):
            target = self.df['label'].iloc[item]
            smiles= self.df['smiles'].iloc[item]
            benzene_positions = []
            
            molecule = Chem.MolFromSmiles(smiles)
            rings = Chem.GetSymmSSSR(molecule)
            for ring in rings:
                # 检查是否是苯环（6 个碳原子）
                if len(ring) == 6:
                    if all(molecule.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6 for atom_idx in ring):
                        benzene_positions.extend(list(ring))
                    

            # mol_size = (500, 500)  # 图像大小
            # drawer = rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])  # Cairo引擎绘图
            # # 配置绘图选项
            # options = drawer.drawOptions()
            # options.addAtomIndices = True  # 显示原子编号
            # options.useBWAtomPalette()  # 使用黑白配色
            # # 绘制分子
            # benzene_positions = list(set(benzene_positions))
            # rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=benzene_positions)
            # drawer.FinishDrawing()
            # # 保存并显示图像
            # img_bytes = drawer.GetDrawingText()
            # with open('molecule{}.png'.format(item), 'wb') as f:
            #     f.write(img_bytes)

            
            # if molecule.HasSubstructMatch(benzene):
            if len(benzene_positions)>0:
                print("该分子包含苯环")
                if target != 1:
                    print('error')
                    continue
                new_target = 1
            else:
                if target != 0:
                    print('error')
                    continue
                print("该分子不包含苯环")
                new_target = 0
            
            
            ground_truth = [0] * molecule.GetNumAtoms()
            for atom_idx in benzene_positions:
                ground_truth[atom_idx] = 1
            
            ground_truth = torch.tensor(ground_truth,dtype=torch.long)
            # molecule = Chem.AddHs(molecule)
            adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(molecule)
            nodes_num = adjacency_matrix.shape[0]
            node_labels = []
            for atom in molecule.GetAtoms():
                node_labels.append(np.array(unique_atoms_dict[atom.GetSymbol()]))
            
            node_labels = np.array(node_labels).reshape(-1,1)
            x = encoder.fit_transform(node_labels)
            row, col = np.nonzero(adjacency_matrix)
            # 构建 edge_index
            edge_index = np.vstack((row, col))
            x = x.todense()
            x = torch.tensor(x).float()
            edge_index = torch.from_numpy(edge_index).long()
            y = torch.tensor(new_target, dtype=torch.long).float()
            data = Data(x=x, y=y, edge_index=edge_index,ground_truth= ground_truth)
            self.data_list.append(data)
        data, slices = self.collate(self.data_list) #2951
        torch.save((data, slices), self.processed_paths[0])
        # torch.save(self.collate(self.data_list), 'data/Solubility/Solubility.pt')
            

        

    # def __len__(self):
    #     return len(self.data_list)

    # def __getitem__(self, item):
    #     mol = smiles_to_graph(self.df['smiles'].iloc[item])
    #     target = self.df['measured log solubility in mols per litre'].iloc[item]
    #     return mol, torch.tensor(target)


def describe(cfg):
    pd.options.display.precision = 2
    pd.options.display.max_columns = 999
    pd.options.display.expand_frame_repr = False
    target = Path(cfg.target).expanduser().resolve()
    if target.is_dir():
        paths = target.glob('*.pt')
    else:
        paths = [target]
    for p in paths:
        print(f"Loading dataset from: {p}")
        dataset = SolubilityDataset(p)
        print(f"{p.with_suffix('').name.capitalize()} contains:\n"
              f"{dataset.df.drop(columns=['molecules']).describe().transpose()}")


def main():
    from argparse import ArgumentParser
    from config import Config

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    sp_print = subparsers.add_parser('print', help='Print parsed configuration')
    sp_print.add_argument('config', nargs='*')
    sp_print.set_defaults(command=lambda c: print(c.toYAML()))

    sp_describe = subparsers.add_parser('describe', help='Describe existing datasets')
    sp_describe.add_argument('config', nargs='*')
    sp_describe.set_defaults(command=describe)

    args = parser.parse_args()
    cfg = Config.build(*args.config)
    args.command(cfg)


if __name__ == '__main__':
    main()
