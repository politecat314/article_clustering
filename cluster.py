import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

def get_cluster(embeddings, similarity=0.55):
    """
    Cluster embeddings using DBSCAN based on cosine similarity.
    
    This function groups embedding vectors that have high cosine similarity with each other.
    It uses DBSCAN clustering algorithm with a precomputed distance matrix derived from
    cosine similarity values.
    
    Parameters
    ----------
    embeddings : list
        A list of numpy arrays, where each array represents an embedding vector.
        All embedding vectors should have the same dimensionality.
    
    similarity : float, default=0.55
        The cosine similarity threshold for clustering. Embeddings with similarity
        greater than or equal to this value will be considered part of the same cluster.
        Must be between 0 and 1, where higher values create tighter clusters.
    
    Returns
    -------
    dict
        A dictionary mapping cluster labels to lists of indices from the original embeddings list.
        The keys are integer cluster labels assigned by DBSCAN (where -1 represents outliers),
        and the values are lists containing the indices of embeddings belonging to each cluster.
    
    Examples
    --------
    >>> embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.15, 0.25, 0.35]), np.array([0.9, 0.8, 0.7])]
    >>> clusters = get_cluster(embeddings)
    >>> print(clusters)
    {0: [0, 1], 1: [2]}
    """

    embeddings_array = np.vstack(embeddings)

    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_array)

    # Since DBSCAN with precomputed metric requires non-negative values,
    # we need to ensure our distance matrix is properly normalized
    # Cosine similarity ranges from -1 to 1, so we'll convert to a range of 0 to 2
    distance_matrix = 1 - similarity_matrix

    # Ensure no negative values (which can happen due to floating point precision)
    distance_matrix = np.maximum(distance_matrix, 0)

    # Perform DBSCAN clustering
    # With our distance matrix, we need eps=0.55 to represent our similarity threshold of 0.45
    # (distance = 1 - similarity, so similarity of 0.45 means distance of 0.55)
    # min_samples = 1 to ensure every point gets assigned to a cluster if possible
    dbscan = DBSCAN(eps=1-similarity, min_samples=1, metric='precomputed')
    cluster_labels = dbscan.fit_predict(distance_matrix)

    # Group indices by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return clusters