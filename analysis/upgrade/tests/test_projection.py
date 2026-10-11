import numpy as np

from higher_order_v2.projection import incidence_projection, triplet_index


def test_mean_triplet_projection_to_edges_and_nodes():
    triplets = triplet_index(4)
    values = np.arange(1, 5, dtype=float)
    projected = incidence_projection(values, triplets, n_nodes=4)

    assert projected.edge_values.shape == (6,)
    assert projected.node_values.shape == (4,)
    edge_map = {tuple(edge): value for edge, value in zip(projected.edges, projected.edge_values)}
    # Edge (0, 1) occurs in triplets (0,1,2) and (0,1,3): mean (1+2)/2.
    assert edge_map[(0, 1)] == 1.5
    # Node 3 occurs in triplets 013, 023, 123: mean (2+3+4)/3.
    assert projected.node_values[3] == 3.0


def test_signed_and_absolute_projections_remain_distinct():
    triplets = triplet_index(4)
    signed = np.array([1.0, -1.0, 1.0, -1.0])
    signed_projection = incidence_projection(signed, triplets, 4)
    magnitude_projection = incidence_projection(np.abs(signed), triplets, 4)
    assert np.any(np.abs(signed_projection.edge_values) < magnitude_projection.edge_values)

