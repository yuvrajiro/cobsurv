import numba
import numpy as np
from numba import prange


@numba.jit(nopython=True)
def index_calculator_helper(i, pred_l,test_pred ,n_estimators, alpha, epsilon, times , distance_func):
    """
    A helper function for index_calculator, it is used for canculating the index of one row, i.e. one test data
    proximity to all training data



    Parameters
    ----------
    i : int
        The observation index
    pred_l : 3d array
        Predictions of all training data
    test_pred : 3d array
        Predictions of all test data
    n_estimators : int
        number of estimators
    alpha : int
        The number of base estimators to be in consensus
    epsilon : float
        the proximity threshold
    times : 1d array
        The times at which the predictions are made
    distance_func : function
        The distance function

    Returns
    -------
    index_arr_row : 1d array
        The index of the test data to all observation in :math:`D_l` which is in :math:`\epsilon,\alpha)` proximity
    """
    index_arr_row = np.zeros(pred_l.shape[0], dtype=np.bool_)

    flag = False

    for j in prange(pred_l.shape[0]):
        n_estimators_consensus = 0

        for k in range(n_estimators):
            if distance_func(test_pred[i, k, :], pred_l[j, k, :], times) <= epsilon:
                n_estimators_consensus += 1

        if n_estimators_consensus >= alpha:
            index_arr_row[j] = True
            flag = True

    if not flag:
        index_arr_row[:] = True

    return index_arr_row


@numba.jit(nopython=True)
def _predict_probability_at_times(uniq_times, probability, times):
    """
    A helper function for predict_probability_at_times, it is used for
    one dimensional array



    Parameters
    ----------
    uniq_times : 1d array
        times at which probability is given
    probability : 1d array
        probability at uniq_times
    times : 1d array
        compute probability at times
    """
    below_range = times < uniq_times[0]
    above_range = times > uniq_times[-1]
    prob_new = np.empty(len(times), dtype=float)
    prob_new[below_range] = 1.0
    slope = (probability[-1] - 1.0 + 1e-16) / (uniq_times[-1] - 0.0 + 1e-16)
    intercept = 1.0 - slope * 0.0
    # times_original = times
    prob_new[above_range] = slope * times[above_range] + intercept
    times = times[~below_range & ~above_range]
    idx = np.searchsorted(uniq_times, times)
    eps = np.finfo(uniq_times.dtype).eps
    exact = np.absolute(uniq_times[idx] - times) < eps
    idx[~exact] -= 1
    prob_new[~below_range & ~above_range] = probability[idx]
    prob_new[prob_new < 0] = 0
    return prob_new


def predict_probability_at_times(uniq_times, probability, times):
    """
    Predict the probability at times

    Parameters
    ----------
    uniq_times : 1d array
        times at which probability is given
    probability : 1d or 2d array
        probability at uniq_times
    times : 1d array
        compute probability at times
    """
    if len(probability.shape) == 1:
        return _predict_probability_at_times(uniq_times, probability, times)
    else:
        prob_new = np.zeros((probability.shape[0], times.shape[0]))
        for i in range(probability.shape[0]):
            prob_new[i] = _predict_probability_at_times(uniq_times,
                                                        probability[i], times)
        return prob_new


