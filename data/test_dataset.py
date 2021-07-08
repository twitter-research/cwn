import pytest
import os

from data.data_loading import load_graph_dataset
from data.datasets import TUDataset, DummyMolecularDataset, DummyDataset
from data.utils import compute_clique_complex_with_gudhi, compute_ring_2complex
from data.helper_test import compare_complexes, compare_complexes_without_2feats
from definitions import ROOT_DIR

    
def validate_data_retrieval(dataset, graph_list, exp_dim, include_down_adj, ring_size=None):
    
    assert len(dataset) == len(graph_list)
    for i in range(len(graph_list)):
        graph = graph_list[i]
        yielded = dataset[i]
        if ring_size is not None:
            expected = compute_ring_2complex(graph.x, graph.edge_index, None,
                                             graph.num_nodes, y=graph.y,
                                             max_k=ring_size, include_down_adj=include_down_adj,
                                             init_rings=True)
        else:
            expected = compute_clique_complex_with_gudhi(graph.x, graph.edge_index,
                                                     graph.num_nodes, expansion_dim=exp_dim,
                                                     y=graph.y, include_down_adj=include_down_adj)
        compare_complexes(yielded, expected, include_down_adj)
        

@pytest.mark.data
def test_data_retrieval_on_proteins():
    dataset = TUDataset(os.path.join(ROOT_DIR, 'datasets', 'PROTEINS'), 'PROTEINS', max_dim=3,
                        num_classes=2, fold=0, degree_as_tag=False, init_method='sum', include_down_adj=True)
    graph_list, train_ids, val_ids, _, num_classes = load_graph_dataset('PROTEINS', fold=0)
    assert dataset.include_down_adj
    assert dataset.num_classes == num_classes
    validate_data_retrieval(dataset, graph_list, 3, True)
    validate_data_retrieval(dataset[train_ids], [graph_list[i] for i in train_ids], 3, True)
    validate_data_retrieval(dataset[val_ids], [graph_list[i] for i in val_ids], 3, True)
    return


@pytest.mark.data
def test_data_retrieval_on_proteins_with_rings():
    dataset = TUDataset(os.path.join(ROOT_DIR, 'datasets', 'PROTEINS'), 'PROTEINS', max_dim=2,
                        num_classes=2, fold=0, degree_as_tag=False, init_method='sum', include_down_adj=True,
                        max_ring_size=6)
    graph_list, train_ids, val_ids, _, num_classes = load_graph_dataset('PROTEINS', fold=0)
    assert dataset.include_down_adj
    assert dataset.num_classes == num_classes
    # Reducing to val_ids only, to save some time. Uncomment the lines below to test on the whole set
    # validate_data_retrieval(dataset, graph_list, 2, True,  6)
    # validate_data_retrieval(dataset[train_ids], [graph_list[i] for i in train_ids], 2, True, 6)
    validate_data_retrieval(dataset[val_ids], [graph_list[i] for i in val_ids], 2, True, 6)
    

def test_dummy_dataset_data_retrieval():
    
    complexes = DummyDataset.factory()
    dataset = DummyDataset(os.path.join(ROOT_DIR, 'datasets', 'DUMMY'))
    assert len(complexes) == len(dataset)
    for i in range(len(dataset)):
        compare_complexes(dataset[i], complexes[i], True)


def test_dummy_mol_dataset_data_retrieval():
    
    complexes = DummyMolecularDataset.factory(False)
    dataset = DummyMolecularDataset(os.path.join(ROOT_DIR, 'datasets', 'DUMMYM'), False)
    assert len(complexes) == len(dataset)
    for i in range(len(dataset)):
        compare_complexes(dataset[i], complexes[i], True)
        

def test_dummy_mol_dataset_data_retrieval_without_2feats():
    
    complexes = DummyMolecularDataset.factory(True)
    dataset = DummyMolecularDataset(os.path.join(ROOT_DIR, 'datasets', 'DUMMYM'), True)
    assert len(complexes) == len(dataset)
    for i in range(len(dataset)):
        compare_complexes_without_2feats(dataset[i], complexes[i], True)
