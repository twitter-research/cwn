from data.data_loading import load_dataset

# We keep this test commented because it takes a lot of compute but it is useful to run manually
# from time to time.

# def test_zinc_actually_works():
#     dataset1 = load_dataset("ZINC", max_ring_size=7, use_edge_features=True)
#     dataset2 = load_dataset("ZINC", max_ring_size=12, use_edge_features=True)
#
#     expected = 33242
#     total1, total2 = 0, 0
#     for i, _ in enumerate(dataset1):
#         data1, data2 = dataset1[i], dataset2[i]
#         assert data1.chains[0].num_simplices == data2.chains[0].num_simplices
#         assert data1.chains[1].num_simplices == data2.chains[1].num_simplices
#
#         if len(data1.chains) > 2:
#             total1 += data1.chains[2].num_simplices
#         if len(data2.chains) > 2:
#             total2 += data2.chains[2].num_simplices
#         # assert data1.chains[2].num_simplices == data2.chains[2].num_simplices
#         # assert data1.chains[2].face_index.max() == data2.chains[2].face_index.max()
#     print(total1, total2)

