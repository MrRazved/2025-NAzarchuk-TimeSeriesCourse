import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    # INSERT YOUR CODE

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate squared DTW distance with Sakoe-Chiba band constraint.
    This version matches the sktime library's default behavior.
    """
    n = len(ts1)
    m = len(ts2)
    
    w = int(np.floor(r * n))

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        start_j = max(1, i - w)
        end_j = min(m, i + w) + 1

        for j in range(start_j, end_j):
            dist = (ts1[i - 1] - ts2[j - 1]) ** 2
            
            last_min = min(dtw_matrix[i - 1, j],
                           dtw_matrix[i, j - 1],
                           dtw_matrix[i - 1, j - 1])
            
            dtw_matrix[i, j] = dist + last_min

    # Return the squared distance, same as sktime
    dtw_dist = dtw_matrix[n, m]

    return dtw_dist