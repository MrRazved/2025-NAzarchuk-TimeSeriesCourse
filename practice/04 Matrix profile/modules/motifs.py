import numpy as np
from modules.utils import *

def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # Создаем копию матричного профиля, чтобы не изменять исходный
    mp_copy = np.copy(matrix_profile['mp'])
    
    # Получаем длину подпоследовательности и размер зоны исключения из структуры
    m = matrix_profile['m']
    # Если зона исключения не задана в структуре, используем m/2 по умолчанию
    if matrix_profile['excl_zone'] is not None:
        excl_zone = matrix_profile['excl_zone']
    else:
        excl_zone = int(np.ceil(m / 2))

    for i in range(top_k):
        # Находим индекс минимального значения в текущем матричном профиле
        min_idx = np.argmin(mp_copy)
        
        # Получаем расстояние (само минимальное значение)
        min_dist = mp_copy[min_idx]
        
        # Если минимальное значение - бесконечность, значит, мы нашли все возможные мотивы
        if np.isinf(min_dist):
            break
            
        # Находим индекс ближайшего соседа для найденного минимума
        nn_idx = matrix_profile['mpi'][min_idx]
        
        # Сохраняем пару индексов и расстояние. sorted() нужен для единообразия, чтобы меньший индекс всегда был первым.
        motifs_idx.append(sorted([min_idx, nn_idx]))
        motifs_dist.append(min_dist)
        
        # Применяем зону исключения вокруг первого индекса
        apply_exclusion_zone(mp_copy, min_idx, excl_zone, np.inf)
        
        # Применяем зону исключения вокруг второго индекса (его соседа)
        apply_exclusion_zone(mp_copy, nn_idx, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
    }