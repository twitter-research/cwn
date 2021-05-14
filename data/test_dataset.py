import torch
import os

from data.data_loading import load_graph_dataset
from data.datasets import TUDataset, DummyMolecularDataset, DummyDataset
from data.utils import compute_clique_complex_with_gudhi, compute_ring_2complex
from definitions import ROOT_DIR

def compare_complexes(yielded, expected, include_down_adj):
    
    assert yielded.dimension == expected.dimension
    assert torch.equal(yielded.y, expected.y)
    for dim in range(expected.dimension + 1):
        y_chain = yielded.chains[dim]
        e_chain = expected.chains[dim]
        assert y_chain.num_simplices == e_chain.num_simplices
        assert y_chain.num_simplices_up == e_chain.num_simplices_up
        assert y_chain.num_simplices_up == e_chain.num_simplices_up
        assert y_chain.num_simplices_down == e_chain.num_simplices_down, dim
        assert torch.equal(y_chain.x, e_chain.x)
        if dim > 0:
            assert torch.equal(y_chain.face_index, e_chain.face_index)
            if include_down_adj:
                if y_chain.lower_index is None:
                    assert e_chain.lower_index is None
                    assert y_chain.shared_faces is None
                    assert e_chain.shared_faces is None
                else:
                    assert torch.equal(y_chain.lower_index, e_chain.lower_index)
                    assert torch.equal(y_chain.shared_faces, e_chain.shared_faces) 
        else:
            assert y_chain.face_index is None and e_chain.face_index is None
            assert y_chain.lower_index is None and e_chain.lower_index is None
            assert y_chain.shared_faces is None and e_chain.shared_faces is None
        if dim < expected.dimension:
            if y_chain.upper_index is None:
                assert e_chain.upper_index is None
                assert y_chain.shared_cofaces is None
                assert e_chain.shared_cofaces is None
            else:
                assert torch.equal(y_chain.upper_index, e_chain.upper_index)
                assert torch.equal(y_chain.shared_cofaces, e_chain.shared_cofaces)
        else:
            assert y_chain.upper_index is None and e_chain.upper_index is None
            assert y_chain.shared_cofaces is None and e_chain.shared_cofaces is None
    

def compare_complexes_without_2feats(yielded, expected, include_down_adj):
    
    assert yielded.dimension == expected.dimension
    assert torch.equal(yielded.y, expected.y)
    for dim in range(expected.dimension + 1):
        y_chain = yielded.chains[dim]
        e_chain = expected.chains[dim]
        assert y_chain.num_simplices == e_chain.num_simplices
        assert y_chain.num_simplices_up == e_chain.num_simplices_up
        assert y_chain.num_simplices_up == e_chain.num_simplices_up
        assert y_chain.num_simplices_down == e_chain.num_simplices_down, dim
        if dim > 0:
            assert torch.equal(y_chain.face_index, e_chain.face_index)
            if include_down_adj:
                if y_chain.lower_index is None:
                    assert e_chain.lower_index is None
                    assert y_chain.shared_faces is None
                    assert e_chain.shared_faces is None
                else:
                    assert torch.equal(y_chain.lower_index, e_chain.lower_index)
                    assert torch.equal(y_chain.shared_faces, e_chain.shared_faces) 
        else:
            assert y_chain.face_index is None and e_chain.face_index is None
            assert y_chain.lower_index is None and e_chain.lower_index is None
            assert y_chain.shared_faces is None and e_chain.shared_faces is None
        if dim < expected.dimension:
            if y_chain.upper_index is None:
                assert e_chain.upper_index is None
                assert y_chain.shared_cofaces is None
                assert e_chain.shared_cofaces is None
            else:
                assert torch.equal(y_chain.upper_index, e_chain.upper_index)
                assert torch.equal(y_chain.shared_cofaces, e_chain.shared_cofaces)
        else:
            assert y_chain.upper_index is None and e_chain.upper_index is None
            assert y_chain.shared_cofaces is None and e_chain.shared_cofaces is None
        if dim != 2:
            assert torch.equal(y_chain.x, e_chain.x)
        else:
            assert y_chain.x is None and e_chain.x is None
    
    
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