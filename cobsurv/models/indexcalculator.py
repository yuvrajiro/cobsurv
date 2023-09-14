import numpy as np
from joblib import Parallel, delayed

from cobsurv.models.distance_functions import distance_euclidean, distance_cross_entropy, \
    distance_cross_entropy_from_mean, distance_shannon_jensen, distance_kl, distance_max, distance_euler, distance_trapz
from ..utils import index_calculator_helper


class IndexCalculator(object):
    """Class for calculating the index of a given CBR object.

    The index is calculated by checking whether the $(\epsilon , \alpha)$
    proximity is satisfied for each case in the CBR.


    Attributes:
        epsilon: The epsilon value to use for the index calculation.
        alpha: The alpha value to use for the index calculation.
        distance: The distance metric to use for the index calculation.
    """
    def __init__(self,query_pred,pred_l,times):
        """Initializes the IndexCalculator with the given CBR object.

        Args:
            epsilon: The epsilon value to use for the index calculation.
            alpha: The alpha value to use for the index calculation.
            distance: The distance metric to use for the index calculation.It
            is set to "area" by default, but a function can be passed instead.
            other options are "euclidean", "cross_entropy" , a symmetric version
            of KL divergence named "shannon_jensen" , and a logrank statistics.

        """
        self.times = times
        self.n_estimators = pred_l.shape[1]
        assert query_pred.shape[1] == pred_l.shape[1] and query_pred.shape[2] == pred_l.shape[2], \
            "query_pred and pred_l have different dimensions"
        assert query_pred.shape[2] == times.shape[0], \
            "query_pred and times have different dimensions"
        assert query_pred.shape[1] == self.n_estimators, \
            "query_pred and n_estimators have different dimensions"

        self.pred_l = pred_l
        self.query_pred = query_pred

        self.index_arr = np.ones((query_pred.shape[0], pred_l.shape[0]), dtype=np.bool_)



    def hypers(self,epsilon,alpha,distance = "area"):
        self.epsilon = epsilon
        self.alpha = alpha
        if isinstance(distance, str):
            if distance == "trapz":
                self.distance = distance_trapz
            elif distance == "euclidean":
                self.distance = distance_euclidean
            elif distance == "cross_entropy":
                self.distance = distance_cross_entropy
            elif distance == "cross_entropy_from_mean":
                self.distance = distance_cross_entropy_from_mean
            elif distance == "shannon_jensen":
                self.distance = distance_shannon_jensen
            elif distance == "kl":
                self.distance = distance_kl
            elif distance == "max":
                self.distance = distance_max
            elif distance == "euler":
                self.distance = distance_euler
            else:
                raise ValueError("distance is not a valid string")
        elif callable(distance):
            self.distance = distance
        else:
            raise ValueError("distance is not a valid function or string")


    def calculate_index(self):
        """
        For a given set of prediction, this function calculates whether  the prediction is in $(\epsilon,\alpha)$ proximity
            :param test_pred: A prediction by machines
            :return: A boolean array of shape (n_samples,n_samples in $D_l$)
        """

        index = Parallel(n_jobs=-1)(delayed(index_calculator_helper)(i, self.pred_l, self.query_pred, self.n_estimators, self.alpha, self.epsilon,
                                            self.times, self.distance) for i in range(self.query_pred.shape[0]))
        self.index_arr = np.array(index)
        return self.index_arr

    def get_index(self, epsilon , alpha , distance = "area"):
        self.hypers(epsilon,alpha,distance)
        return self.calculate_index()

