import numpy as np


def compute_gcn_norm(graph, gcn_norm=False):
    if gcn_norm:
        degree = graph.indegree()
        norm = degree.astype(np.float32)
        norm = np.clip(norm, 1.0, np.max(norm))
        norm = np.power(norm, -0.5)
        norm = np.reshape(norm, [-1, 1])
    else:
        norm = None
    return norm
