import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt


def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0,1:] = np.inf
    dtw[1:,0] = np.inf
    eps = 1e-4
    for i in range(1,costs.shape[0]):
        for j in range(1,costs.shape[1]):
            dtw[i,j] = costs[i,j] + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return dtw


def compute_euclidean_distance(ecog, mspec, n_components=10):
    # Apply PCA to reduce dimensions
    pca_ecog = PCA(n_components=min(n_components, ecog.shape[1]))
    pca_mspec = PCA(n_components=min(n_components, mspec.shape[1]))
    
    ecog_reduced = pca_ecog.fit_transform(ecog)
    mspec_reduced = pca_mspec.fit_transform(mspec)
    
    # Compute distance matrix
    distance_matrix = euclidean_distances(ecog_reduced, mspec_reduced)
    return distance_matrix


def align_from_distances(ecog, mspec, debug=False):
    distance_matrix = compute_euclidean_distance(ecog, mspec)
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i-1,j),(i,j-1),(i-1,j-1)], key=lambda x: dtw[x[0],x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)),results] = 1
        # plt.matshow(visual)
        # plt.show()

    return results