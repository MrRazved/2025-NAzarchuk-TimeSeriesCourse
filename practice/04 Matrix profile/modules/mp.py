import numpy as np
import pandas as pd
import math

import stumpy
from stumpy import config


def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None, ts2: np.ndarray = None):
    """
    Compute the matrix profile

    Parameters
    ----------
    ts1: the first time series
    m: the subsequence length
    exclusion_zone: exclusion zone
    ts2: the second time series

    Returns
    -------
    output: the matrix profile structure
            (matrix profile, matrix profile index, subsequence length, exclusion zone, the first and second time series)
    """
    
    if ts2 is None:
        # Это случай, когда мы ищем матричный профиль для одного ряда (self-join).
        # ignore_trivial=True автоматически исключает тривиальные совпадения (подпоследовательность с самой собой).
        mp = stumpy.stump(ts1, m=m, ignore_trivial=True)
    else:
        # Это случай для двух разных рядов (A-B join), понадобится в Задаче 5.
        # Тривиальных совпадений здесь быть не может, поэтому ignore_trivial=False.
        mp = stumpy.stump(T_A=ts1, m=m, T_B=ts2, ignore_trivial=False)

    return {'mp': mp[:, 0],
            'mpi': mp[:, 1],
            'm' : m,
            'excl_zone': exclusion_zone,
            'data': {'ts1' : ts1, 'ts2' : ts2}
            }