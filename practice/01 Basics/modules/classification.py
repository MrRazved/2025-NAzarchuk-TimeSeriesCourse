import numpy as np
from collections import Counter

from .metrics import ED_distance, norm_ED_distance, DTW_distance
from .utils import z_normalize

default_metrics_params = {'euclidean': {'normalize': True},
                         'dtw': {'normalize': True, 'r': 0.05}
                         }

class TimeSeriesKNN:
    """
    KNN Time Series Classifier
    """
    
    def __init__(self, n_neighbors: int = 3, metric: str = 'euclidean', metric_params: dict | None = None) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = default_metrics_params[metric].copy()
        if metric_params is not None:
            self.metric_params.update(metric_params)

        self.X_train = None
        self.Y_train = None 

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> 'TimeSeriesKNN':
        self.X_train = X_train
        self.Y_train = Y_train 
        return self

    def _distance(self, x_train: np.ndarray, x_test: np.ndarray) -> float:
        use_norm = self.metric_params.get('normalize', False)

        if self.metric == 'euclidean':
            if use_norm:
                dist = norm_ED_distance(x_train, x_test)
            else:
                dist = ED_distance(x_train, x_test)
        elif self.metric == 'dtw':
            if use_norm:
                x_train_norm = z_normalize(x_train)
                x_test_norm = z_normalize(x_test)
                dist = DTW_distance(x_train_norm, x_test_norm)
            else:
                dist = DTW_distance(x_train, x_test)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return dist

    def _find_neighbors(self, x_test: np.ndarray) -> list:
        distances = []
        for i in range(len(self.X_train)):
            dist = self._distance(self.X_train[i], x_test)
            
            distances.append((dist, self.Y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.n_neighbors]
        return neighbors

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = []
        
        for test_sample in X_test:
            neighbors = self._find_neighbors(test_sample)
            neighbor_labels = [neighbor[1] for neighbor in neighbors]
            most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
            y_pred.append(most_common_label)
        
        return np.array(y_pred)

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    if len(y_true) == 0:
        return 0.0
    score = np.sum(y_true == y_pred) / len(y_true)
    return score