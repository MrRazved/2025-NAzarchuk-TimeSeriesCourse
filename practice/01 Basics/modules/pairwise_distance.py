import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:

        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """

        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dict_func: function reference
        """

        #dist_func = None

        # INSERT YOUR CODE

        #return dist_func

        if self.metric == 'euclidean':
            return ED_distance
        elif self.metric == 'dtw':
            return DTW_distance
        else:
        
            raise ValueError("Unknown metric. Use 'euclidean' or 'dtw'.")


    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        
        Returns
        -------
        matrix_values: distance matrix
        """
        
        if self.metric == 'euclidean' and self.is_normalize:
            dist_func = norm_ED_distance
            data_to_process = input_data
        else:
        
            dist_func = self._choose_distance()
        
        
        if self.is_normalize:
            data_to_process = z_normalize(input_data)
        else:
            data_to_process = input_data

  
    
        matrix_shape = (data_to_process.shape[0], data_to_process.shape[0])
        matrix_values = np.zeros(shape=matrix_shape)
        
        k = data_to_process.shape[0]
        
        for i in range(k):
            for j in range(i + 1, k):
                dist = dist_func(data_to_process[i], data_to_process[j])
                matrix_values[i, j] = dist
                matrix_values[j, i] = dist
            
        return matrix_values