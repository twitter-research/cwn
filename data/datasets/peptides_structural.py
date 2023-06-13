#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:37:42 2023

@author: renz
"""

import hashlib
import os.path as osp
import os
import pickle
import shutil

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, download_url
from torch_geometric.data import InMemoryDataset
from data.utils import convert_graph_dataset_with_rings
from data.datasets import InMemoryComplexDataset
from tqdm import tqdm


class PeptidesStructuralDataset(InMemoryComplexDataset):
    """
    PyG dataset of 15,535 small peptides represented as their molecular
    graph (SMILES) with 11 regression targets derived from the peptide's
    3D structure.

    The original amino acid sequence representation is provided in
    'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
    of the dataset file, but not used here as any part of the input.

    The 11 regression targets were precomputed from molecule XYZ:
        Inertia_mass_[a-c]: The principal component of the inertia of the
            mass, with some normalizations. Sorted
        Inertia_valence_[a-c]: The principal component of the inertia of the
            Hydrogen atoms. This is basically a measure of the 3D
            distribution of hydrogens. Sorted
        length_[a-c]: The length around the 3 main geometric axis of
            the 3D objects (without considering atom types). Sorted
        Spherocity: SpherocityIndex descriptor computed by
            rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
        Plane_best_fit: Plane of best fit (PBF) descriptor computed by
            rdkit.Chem.rdMolDescriptors.CalcPBF
    Args:
        root (string): Root directory where the dataset should be saved.
        smiles2graph (callable): A callable function that converts a SMILES
            string into a graph object. We use the OGB featurization.
            * The default smiles2graph requires rdkit to be installed *
    """
    def __init__(self, root, max_ring_size, smiles2graph=smiles2graph,
                 transform=None, pre_transform=None, pre_filter=None, 
                 include_down_adj=False, init_method='sum', n_jobs=2):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'peptides-structural')

        self.url = 'https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1'
        self.version = '9786061a34298a0684150f2e4ff13f47'  # MD5 hash of the intended dataset file
        self.url_stratified_split = 'https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1'
        self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)

        self.name = 'peptides_structural'
        self._max_ring_size = max_ring_size
        self._use_edge_features = True
        self._n_jobs = n_jobs
        super(PeptidesStructuralDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                          max_dim=2, init_method=init_method, include_down_adj=include_down_adj,
                                          cellular=True, num_classes=1)
        
        self.data, self.slices, idx, self.num_tasks = self.load_dataset()
        self.train_ids = idx['train']
        self.val_ids = idx['val']
        self.test_ids = idx['test']
        self.num_node_type = 9
        self.num_edge_type = 3

    @property
    def raw_file_names(self):
        return 'peptide_structure_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return [f'{self.name}_complex.pt', f'{self.name}_idx.pt', f'{self.name}_tasks.pt']


    @property
    def processed_dir(self):
        """Overwrite to change name based on edge and simple feats"""
        directory = super(PeptidesStructuralDataset, self).processed_dir
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        return directory + suffix1 + suffix2


    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
            
            old_split_file = osp.join(self.root,
                              "splits_random_stratified_peptide_structure.pickle?dl=1")
            new_split_file = osp.join(self.root,
                              "splits_random_stratified_peptide_structure.pickle")
            old_df_name = osp.join(self.raw_dir,
                                       'peptide_structure_dataset.csv.gz?dl=1')
            new_df_name = osp.join(self.raw_dir,
                                       'peptide_structure_dataset.csv.gz')
            os.rename(old_split_file, new_split_file)
            os.rename(old_df_name, new_df_name)
        else:
            print('Stop download.')
            exit(-1)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        tasks = torch.load(self.processed_paths[2])
        return data, slices, idx, tasks

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir,
                                       'peptide_structure_dataset.csv.gz'))
        smiles_list = data_df['smiles']
        target_names = ['Inertia_mass_a', 'Inertia_mass_b', 'Inertia_mass_c',
                        'Inertia_valence_a', 'Inertia_valence_b',
                        'Inertia_valence_c', 'length_a', 'length_b', 'length_c',
                        'Spherocity', 'Plane_best_fit']
        # Normalize to zero mean and unit standard deviation.
        data_df.loc[:, target_names] = data_df.loc[:, target_names].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0)

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][target_names]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(
                torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([y])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        split_idx = self.get_idx_split()
        
        # NB: the init method would basically have no effect if 
        # we use edge features and do not initialize rings. 
        print(f"Converting the {self.name} dataset to a cell complex...")
        complexes, _, _ = convert_graph_dataset_with_rings(
            data_list,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_method=self._init_method,
            init_edges=self._use_edge_features,
            init_rings=False,
            n_jobs=self._n_jobs)
        
        print(f'Saving processed dataset in {self.processed_paths[0]}...')
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
        
        print(f'Saving idx in {self.processed_paths[1]}...')
        torch.save(split_idx, self.processed_paths[1])
        
        print(f'Saving num_tasks in {self.processed_paths[2]}...')
        torch.save(11, self.processed_paths[2])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root,
                              "splits_random_stratified_peptide_structure.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        split_dict['valid'] = split_dict['val']
        return split_dict


def load_pep_s_graph_dataset(root):
    raw_dir = osp.join(root, 'raw')
    data_df = pd.read_csv(osp.join(raw_dir,
                                    'peptide_structure_dataset.csv.gz'))
    smiles_list = data_df['smiles']
    target_names = ['Inertia_mass_a', 'Inertia_mass_b', 'Inertia_mass_c',
                    'Inertia_valence_a', 'Inertia_valence_b',
                    'Inertia_valence_c', 'length_a', 'length_b', 'length_c',
                    'Spherocity', 'Plane_best_fit']
    # Normalize to zero mean and unit standard deviation.
    data_df.loc[:, target_names] = data_df.loc[:, target_names].apply(
        lambda x: (x - x.mean()) / x.std(), axis=0)

    print('Converting SMILES strings into graphs...')
    data_list = []
    for i in tqdm(range(len(smiles_list))):
        data = Data()

        smiles = smiles_list[i]
        y = data_df.iloc[i][target_names]
        graph = smiles2graph(smiles)

        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])

        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(
            torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
            torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        data.y = torch.Tensor([y])

        data_list.append(data)

    dataset = InMemoryDataset.collate(data_list)

    #get split file
    split_file = osp.join(root,
                              "splits_random_stratified_peptide_structure.pickle")
    with open(split_file, 'rb') as f:
        splits = pickle.load(f)
    split_dict = replace_numpy_with_torchtensor(splits)
    split_dict['valid'] = split_dict['val']

    return dataset, split_dict['train'], split_dict['valid'], split_dict['test']