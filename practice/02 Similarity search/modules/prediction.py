import numpy as np
import math

from modules.bestmatch import UCR_DTW, topK_match
import mass_ts as mts


default_match_alg_params = {
    'UCR-DTW': {
        'topK': 3,
        'r': 0.05,
        'excl_zone_frac': 1,
        'is_normalize': True
    },
    'MASS': {
        'topK': 3,
        'excl_zone_frac': 1
    }
}


class BestMatchPredictor:
    """
    Predictor based on best match algorithm
    """

    def __init__(self, h: int = 1, match_alg: str = 'UCR-DTW', match_alg_params: dict | None = None, aggr_func: str = 'average') -> None:
        """ 
        Constructor of class BestMatchPredictor

        Parameters
        ----------    
        h: prediction horizon
        match_algorithm: name of the best match algorithm
        match_algorithm_params: input parameters for the best match algorithm
        aggr_func: aggregate function
        """

        self.h: int = h
        self.match_alg: str = match_alg
        self.match_alg_params: dict | None = default_match_alg_params[match_alg].copy()
        if match_alg_params is not None:
            self.match_alg_params.update(match_alg_params)
        self.agg_func: str = aggr_func


    def _calculate_predict_values(self, topK_subs_predict_values: np.array) -> np.ndarray:
        """
        Calculate the future values of the time series using the aggregate function

        Parameters
        ----------
        topK_subs_predict_values: values of time series, which are located after topK subsequences

        Returns
        -------
        predict_values: prediction values
        """

        match self.agg_func:
            case 'average':
                predict_values = topK_subs_predict_values.mean(axis=0).round()
            case 'median':
                predict_values = topK_subs_predict_values.median(axis=0).round()
            case _:
                raise NotImplementedError
        
        return predict_values


    def predict(self, ts: np.ndarray, query: np.ndarray) -> np.array:
  
        predict_values = np.zeros((self.h,))
        m = len(query)

        # --- 1. Находим topK похожих подпоследовательностей ---
        best_matches = {}
        if self.match_alg == 'UCR-DTW':
            finder = UCR_DTW(**self.match_alg_params)
            best_matches = finder.perform(ts, query)
        elif self.match_alg == 'MASS':
            dist_profile = mts.mass(ts, query)
            excl_zone = math.ceil(m * self.match_alg_params['excl_zone_frac'])
            best_matches = topK_match(dist_profile, excl_zone, topK=self.match_alg_params['topK'])
        else:
            raise NotImplementedError

        match_indices = best_matches.get('indices', [])
        if not match_indices:
            # Если ничего не найдено, возвращаем нули
            return predict_values

        # --- 2. Собираем "будущие" значения, следующие за найденными совпадениями ---
        topK_subs_predict_values = []
        for idx in match_indices:
            # Начало "будущего" - это конец найденной подпоследовательности
            start = idx + m
            end = start + self.h
            # Убедимся, что не выходим за пределы временного ряда
            if end <= len(ts):
                future_segment = ts[start:end]
                topK_subs_predict_values.append(future_segment)
        
        if not topK_subs_predict_values:
            return predict_values

        # --- 3. Агрегируем (усредняем) "будущие" значения ---
        predict_values = self._calculate_predict_values(np.array(topK_subs_predict_values))

        return predict_values