import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations

def sinne_scores(L, D, q, params):
    """
    SiNNE score computation for subspaces.
    
    Args:
    L (numpy.ndarray): List of subspaces (each is a row of indices)
    D (numpy.ndarray): Dataset (n x d)
    q (numpy.ndarray): Query instance (1 x d)
    params (list): List of parameters [psi, t]
    
    Returns:
    numpy.ndarray: Scores for each subspace
    """
    psi, t = params
    n = L.shape[0]
    scores = np.zeros(n)
    if D.shape[0] < psi:
        psi = int(D.shape[0]/2)


    for i in range(n):
        s = L[i, :]
        data = D[:, s]
        query = q[s]
        fm = np.zeros((t, psi), dtype=bool)

        for j in range(t):
            # Randomly sample `psi` points
            spheres = data[np.random.choice(data.shape[0], psi, replace=False), :]

            # Compute pairwise squared Euclidean distance matrix
            dist = cdist(spheres, spheres, 'sqeuclidean')

            # Avoid self-distance by setting diagonal elements to inf
            np.fill_diagonal(dist, np.inf)

            # Compute radii as the second smallest distance in each row
            radii = np.min(dist, axis=1)

            # Calculate squared Euclidean distance from the query point to each sphere center
            dist_query = cdist(query.reshape(1, -1), spheres, 'sqeuclidean').flatten()

            # Check if the query point is inside the sphere
            fm[j, :] = dist_query <= radii

        # The score is the fraction of successful trials where the query point falls inside a sphere
        scores[i] = np.mean(np.any(fm, axis=1))

    return scores

def update_top_subspaces(S, S_scores, L, L_scores, k, W):
    """
    Update the top-k subspaces based on their scores.
    
    Args:
    S (list): List of subspaces
    S_scores (numpy.ndarray): List of corresponding scores
    L (numpy.ndarray): New set of subspaces
    L_scores (numpy.ndarray): Corresponding scores for new subspaces
    k (int): Number of top subspaces to return
    W (int): Beam width
    
    Returns:
    tuple: Updated subspaces, their scores, and the updated subspaces list
    """
    S_extended = S + list(L)
    S_scores_extended = np.concatenate([S_scores, L_scores])

    if len(S_scores_extended) > 0:
        sorted_indices_S = np.argsort(S_scores_extended)[-k:]
        S_new = [S_extended[i] for i in sorted_indices_S]
        S_scores_new = S_scores_extended[sorted_indices_S]
    else:
        S_new, S_scores_new = [], []
    
    # Keep the top-W subspaces at each depth
    if len(L) > 0:
        sorted_indices_L = np.argsort(L_scores)[-min(W, len(L)):]
        L_new = [L[i] for i in sorted_indices_L]
    else:
        L_new = []

    return S_new, S_scores_new, L_new

def beam2(q, L_max, D, W, k):
    """
    Beam Algorithm for Characteristic Mining using SiNNE score.
    
    Args:
    q (numpy.ndarray): Query instance (1 x d vector)
    L_max (int): Maximum search depth
    D (numpy.ndarray): Dataset (n x d matrix, where n is #instances and d is #attributes)
    W (int): Beam width (number of top subspaces to keep at each depth)
    k (int): Number of top subspaces to return
    
    Returns:
    list: Set of top-k subspaces
    """
    n, d = D.shape
    L_max = min(L_max, d)

    # Initialize top-k subspaces and scores
    S = [None] * k
    S_scores = -np.inf * np.ones(k)  # Top-k scores
    psi = 32
    if n < psi:
        psi = int(n/2)

    # Step 1: Single attribute subspaces
    L = np.arange(d).reshape(-1, 1)
    L_scores = sinne_scores(L, D, q, [psi, 100])  # Psi=32, t=100
    S, S_scores, _ = update_top_subspaces(S, S_scores, L, L_scores, k, W)

    if d < 2:
        return S

    # Step 2: Pairwise attribute subspaces
    L = np.array([list(pair) for pair in combinations(range(d), 2)])
    L_scores = sinne_scores(L, D, q, [32, 100])
    S, S_scores, L = update_top_subspaces(S, S_scores, L, L_scores, k, W)

    # Step 3: Higher-dimensional subspaces
    for l in range(3, L_max + 1):
        L_new = []
        for s in L:
            attributes = set(range(d)) - set(s)  # Get the remaining attributes
            for a in attributes:
                L_new.append(list(s) + [a])  # Generate new subspaces by adding a new attribute

        L_new = np.array(L_new)
        
        # Ensure that all indices in L_new are within the valid range
        L_new = L_new[L_new.max(axis=1) < d]  # Filter invalid subspaces
        
        L_scores = sinne_scores(L_new, D, q, [32, 100])
        S, S_scores, L = update_top_subspaces(S, S_scores, L_new, L_scores, k, W)

    return S
