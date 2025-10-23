import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    # Создаем копию матричного профиля, чтобы не изменять исходный
    mp_copy = np.copy(matrix_profile['mp'])

    # Получаем необходимые параметры из структуры
    m = matrix_profile['m']
    if matrix_profile['excl_zone'] is not None:
        excl_zone = matrix_profile['excl_zone']
    else:
        excl_zone = int(np.ceil(m / 2))

    for i in range(top_k):
        # Находим индекс максимального значения в текущем матричном профиле
        max_idx = np.argmax(mp_copy)
        
        # Получаем расстояние (само максимальное значение)
        max_dist = mp_copy[max_idx]
        
        # Если максимальное значение - бесконечность (или 0), значит, мы нашли все диссонансы
        if np.isinf(max_dist) or max_dist == 0:
            break
        
        # Сохраняем индекс диссонанса, его расстояние и индекс его "ближайшего" (очень далекого) соседа
        discords_idx.append(max_idx)
        discords_dist.append(max_dist)
        discords_nn_idx.append(matrix_profile['mpi'][max_idx])
        
        # Применяем зону исключения вокруг найденного диссонанса.
        # Заполняем нулями, так как мы ищем максимум.
        apply_exclusion_zone(mp_copy, max_idx, excl_zone, 0)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
