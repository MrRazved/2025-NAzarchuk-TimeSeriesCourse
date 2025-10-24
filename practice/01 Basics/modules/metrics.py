import numpy as np
from .utils import z_normalize

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
    
    ed_dist = 0

    # INSERT YOUR CODE
    
    dist = np.sqrt(np.sum((ts1 - ts2)**2))
    return dist


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

    ts1_norm = z_normalize(ts1)
    ts2_norm = z_normalize(ts2)
    
    
    dist = ED_distance(ts1_norm, ts2_norm)
    
    return dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    #dtw_dist = 0

    # INSERT YOUR CODE

    #return dtw_dist
    n = len(ts1)
    m = len(ts2)
    
    
    dtw_matrix = np.zeros((n + 1, m + 1))
    
   
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    
 
    for i in range(1, n + 1):
        for j in range(1, m + 1):

            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            
 
            last_min = min(dtw_matrix[i-1, j],    
                           dtw_matrix[i, j-1],    
                           dtw_matrix[i-1, j-1])  
            
            dtw_matrix[i, j] = cost + last_min
            
 
    return dtw_matrix[n, m]
