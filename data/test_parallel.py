import pytest

from data.dummy_complexes import get_mol_testing_complex_list, convert_to_graph
from data.utils import convert_graph_dataset_with_rings
from data.test_dataset import compare_complexes


@pytest.mark.slow
def test_parallel_conversion_returns_same_order():
    
    complexes = get_mol_testing_complex_list()
    graphs = [convert_to_graph(comp) for comp in complexes]
 
    seq_complexes, _, _ = convert_graph_dataset_with_rings(graphs, init_rings=True, n_jobs=1)
    par_complexes, _, _ = convert_graph_dataset_with_rings(graphs, init_rings=True, n_jobs=2)
    
    for comp_a, comp_b in zip(seq_complexes, par_complexes):
        compare_complexes(comp_a, comp_b, True)
