import numba
import numpy as np


EPSILON = 1e-6

@numba.jit(nopython=True)
def distance_trapz(prob1, prob2, times):
    """
    Calculates the area between two curves estimated by the trapezoidal rule
    Parameters
    ----------
    prob1 : 1d array
        The first curve
    prob2 : 1d array
        The second curve
    times : 1d array
        The times at which the curves are evaluated

    Returns
    -------
    area : float
        The area between the curves
    """
    area = np.trapz(np.abs(prob1 - prob2), times)
    return area/((times[-1] - times[0]) + EPSILON)

@numba.jit(nopython=True)
def distance_euler(prob1, prob2, times):
    """
    Calculates the area between two curves estimated by the euler rule
    Parameters
    ----------
    prob1 : 1d array
        The first curve
    prob2 : 1d array
        The second curve
    times : 1d array
        The times at which the curves are evaluated

    Returns
    -------
    area : float
        The area between the curves
    """

    area = np.sum((np.abs(prob1 - prob2)) * np.diff(add_zero(times)))
    return area/((times[-1] - times[0]) + EPSILON)

@numba.jit(nopython=True)
def add_zero(times):
    """
    This function concatenates 0 to the times array
    :param times:
    :return: return the times array with 0 concatenated
    """
    return np.concatenate((np.array([0]),times))



@numba.jit(nopython=True)
def distance_euclidean(prob1, prob2, times = None):
    """
    Calculates the euclidean distance between two curves at respective timepoints

    Parameters
    ----------
    prob1 : 1d array
        The first curve
    prob2 : 1d array
        The second curve
    times : None
        times is not needed for this distance

    Returns
    -------
    euclidean distance : float
        The euclidean distance between the curves
    """
    print("Using Euler distance")
    return np.sqrt(np.sum((prob1 - prob2)**2))

@numba.jit(nopython=True)
def distance_cross_entropy(prob1, prob2, times = None):
    """
    Calculates the cross entropy between two curves at respective timepoints

    Parameters
    ----------
    prob1 : 1d array
        The first curve
    prob2 : 1d array
        The second curve
    times : None
        times is not needed for this distance

    Returns
    -------
    cross entropy : float
        The cross entropy between the curves

    """
    return -np.sum(prob1*np.log(prob2 + EPSILON))


@numba.jit(nopython=True)
def distance_cross_entropy_from_mean(prob1, prob2, times = None):
    """
    The cross entropy between two curves and the mean of the two curves

    Parameters
    ----------
    prob1 : 1d array
        The first curve
    prob2 : 1d array
        The second curve
    times : None
        times is not needed for this distance

    Returns
    -------
    cross entropy : float
        The cross entropy between the curves and the mean of the curves
    """
    mean_prob = (prob1 + prob2)/2
    return distance_cross_entropy(prob1, mean_prob) + distance_cross_entropy(prob2, mean_prob)


@numba.jit(nopython=True)
def distance_kl(prob1, prob2 , times= None):
    """
    KL divergence of second probability from first probability

    Parameters
    ----------
    prob1 : 1d array
        The first curve
    prob2 : 1d array
        The second curve
    times : None
        times is not needed for this distance

    Returns
    -------
    KL divergence : float
        The KL divergence of the second probability from the first probability
    """
    return np.sum(prob1*np.log(prob1/(prob2 + EPSILON) + EPSILON))

@numba.jit(nopython=True)
def distance_shannon_jensen(prob1, prob2, times = None):
    """
    The symmetric version of KL divergence

    Parameters
    ----------
    prob1 : 1d array
        The first curve
    prob2 : 1d array
        The second curve
    times : None
        times is not needed for this distance

    Returns
    -------
    KL divergence : float
        The symmetric version of KL divergence
    """
    mean_prob = (prob1 + prob2)/2
    return distance_kl(prob1, mean_prob,times = None) + distance_kl(prob2, mean_prob , times = None)

@numba.jit(nopython=True)
def distance_max(prob1, prob2, times = None):
    """
    The maximum difference between two curves

    Parameters
    ----------
    prob1 : 1d array
        The first curve
    prob2 : 1d array
        The second curve
    times : None
        times is not needed for this distance

    Returns
    -------
    max difference : float
        The maximum difference between the two curves

    """
    return np.max(np.abs(prob1 - prob2))


