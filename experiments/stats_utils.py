import numpy as np
from scipy.sparse import csgraph

def inter_intra_counts(S, groups):
    groups = np.asarray(groups)
    intra = {g: 0 for g in np.unique(groups)}
    inter = {g: 0 for g in np.unique(groups)}

    rows, cols = S.nonzero()
    for r, c in zip(rows, cols):
        if r >= c:  # upper triangle only
            continue
        gr, gc = groups[r], groups[c]
        if gr == gc:
            intra[gr] += 1          # edge inside group gr
        else:
            inter[gr] += 1          # edge from gr to another group
            inter[gc] += 1          # same edge seen from gc

    return intra, inter

def lcc_size_numpy(A):
    n_components, labels = csgraph.connected_components(A, directed=False)
    sizes = np.bincount(labels)
    return int(sizes.max()), int(n_components)


def characteristic_path_length(A):
    # work on largest connected component
    n_components, labels = csgraph.connected_components(A, directed=False)
    if n_components > 1:
        # pick largest component
        largest = np.argmax(np.bincount(labels))
        keep = np.flatnonzero(labels == largest)
        A = A.tocsr()[keep][:, keep]

    dist = csgraph.shortest_path(A, directed=False, unweighted=True)
    # upper triangle, exclude zeros on diagonal
    i, j = np.triu_indices_from(dist, k=1)
    return dist[i, j].mean()


