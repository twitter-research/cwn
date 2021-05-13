import torch
from data.data_loading import DataLoader, load_dataset, load_graph_dataset
from data.utils import compute_clique_complex_with_gudhi

def validate_data_retrieval(dataset, graph_list):
    
    assert len(dataset) == len(graph_list)
    for i in range(len(graph_list)):
        graph = graph_list[i]
        yielded = dataset[i]
        expected = compute_clique_complex_with_gudhi(graph.x, graph.edge_index,
                                                     graph.num_nodes, expansion_dim=2,
                                                     y=graph.y)
        assert yielded.dimension == expected.dimension
        assert torch.equal(yielded.y, expected.y)
        for dim in range(expected.dimension + 1):
            y_chain = yielded.chains[dim]
            e_chain = expected.chains[dim]
            assert y_chain.num_simplices == e_chain.num_simplices
            assert y_chain.num_simplices_up == e_chain.num_simplices_up
            assert y_chain.num_simplices_up == e_chain.num_simplices_up
            assert y_chain.num_simplices_down == e_chain.num_simplices_down
            assert torch.equal(y_chain.x, e_chain.x)
            if dim > 0:
                assert torch.equal(y_chain.face_index, e_chain.face_index)
            else:
                assert y_chain.face_index is None and e_chain.face_index is None
            if dim < expected.dimension:
                assert torch.equal(y_chain.upper_index, e_chain.upper_index)
                assert torch.equal(y_chain.shared_cofaces, e_chain.shared_cofaces)
            else:
                assert y_chain.upper_index is None and e_chain.upper_index is None
                assert y_chain.shared_cofaces is None and e_chain.shared_cofaces is None

                
def test_data_retrieval_on_proteins():
    dataset = load_dataset('PROTEINS', max_dim=2, fold=0, init_method='sum')
    graph_list, train_ids, val_ids, _, num_classes = load_graph_dataset('PROTEINS', fold=0)
    assert dataset.num_classes == num_classes
    validate_data_retrieval(dataset, graph_list)
    validate_data_retrieval(dataset[train_ids], [graph_list[i] for i in train_ids])
    validate_data_retrieval(dataset[val_ids], [graph_list[i] for i in val_ids])    
    return