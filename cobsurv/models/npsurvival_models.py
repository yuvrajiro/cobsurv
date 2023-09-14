"""
The credits for the following code, goes to the author of the following repository:
- https://github.com/georgehc/npsurvival/tree/master

We do not claim the authorship, and we have only modified the code to suit our needs.

The code is attached in package, because the original repository may change in future
and we want to keep the code intact.

"""






















import numpy as np
from sklearn.neighbors import NearestNeighbors


class BasicSurvival():
    def __init__(self):
        self.tree = None

    def fit(self, y):
        self.tree = _fit_leaf(y)

    def predict_surv(self, times, presorted_times=False,
                      limit_from_left=False):
        """
        Computes the Kaplan-Meier survival probability function estimate at
        user-specified times.

        Parameters
        ----------
        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the survival
            probability function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        Returns
        -------
        output : 1D numpy array
            Survival probability function evaluated at each of the times
            specified in `times`.
        """
        return _predict_leaf(self.tree, 'surv', times, presorted_times,
                             limit_from_left)

    def predict_cum_haz(self, times, presorted_times=False,
                        limit_from_left=False):
        """
        Computes the Nelson-Aalen cumulative hazard function estimate at
        user-specified times.

        Parameters
        ----------
        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the
            cumulative hazard function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        Returns
        -------
        output : 1D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times`.
        """
        return _predict_leaf(self.tree, 'cum_haz', times, presorted_times,
                             limit_from_left)


class KNNSurvival():
    def __init__(self, *args, **kwargs):
        """
        Arguments are the same as for `sklearn.neighbors.NearestNeighbors`.
        The simplest usage of this class is to use a single argument, which is
        `n_neighbors` for the number of nearest neighbors (Euclidean distance
        is assumed in this case). If you want to parallelize across different
        search queries, use the `n_jobs` keyword parameter (-1 to use all
        cores). To use other distances and for other details, please refer to
        the documentation for sklearn's `NearestNeighbors` class.

        *Important:* The prediction methods for this class use unweighted
        k-nearest neighbors, where "k" is set equal to the `n_neighbors`
        parameter.
        """
        self.NN_index_args = args
        self.NN_index_kwargs = kwargs
        self.NN_index = None

    def fit(self, X, y):
        """
        Constructs a nearest-neighbor index given training data (so that for
        a future data point, we can use the nearest-neighbor index to quickly
        find what the closest training data are to the future point).

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        Returns
        -------
        None
        """
        self.train_y = y
        self.NN_index = NearestNeighbors(*self.NN_index_args,
                                         **self.NN_index_kwargs)
        self.NN_index.fit(X)

    def predict_surv(self, X, times, presorted_times=False,
                      limit_from_left=False, n_neighbors=None):
        """
        Computes the k-NN Kaplan-Meier survival probability function estimate
        at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the survival
            probability function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                           return_distance=False)
        train_y = self.train_y
        return np.array([_predict_leaf(_fit_leaf(train_y[idx]), 'surv', times,
                                       presorted_times, limit_from_left)
                         for idx in indices])

    def predict_cum_haz(self, X, times, presorted_times=False,
                        limit_from_left=False, n_neighbors=None):
        """
        Computes the k-NN Nelson-Aalen cumulative hazard function estimate at
        user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the
            cumulative hazard function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                           return_distance=False)
        train_y = self.train_y
        return np.array([_predict_leaf(_fit_leaf(train_y[idx]), 'cum_haz',
                                       times, presorted_times, limit_from_left)
                         for idx in indices])


class KNNWeightedSurvival():
    def __init__(self, *args, **kwargs):
        """
        Arguments are the same as for `sklearn.neighbors.NearestNeighbors`.
        The simplest usage of this class is to use a single argument, which is
        `n_neighbors` for the number of nearest neighbors (Euclidean distance
        is assumed in this case). If you want to parallelize across different
        search queries, use the `n_jobs` keyword parameter (-1 to use all
        cores). To use other distances and for other details, please refer to
        the documentation for sklearn's `NearestNeighbors` class.

        *Important:* The prediction methods for this class use weighted
        k-nearest neighbors, where "k" is set equal to the `n_neighbors`
        parameter. The weights are specified through a kernel function K. In
        particular, the i-th nearest neighbor X_i for a test point x is given a
        weight of:
            K( (distance between x and X_i) / (distance between x and X_k) ).
        """
        self.NN_index_args = args
        self.NN_index_kwargs = kwargs
        self.NN_index = None

    def fit(self, X, y):
        """
        Constructs a nearest-neighbor index given training data (so that for
        a future data point, we can use the nearest-neighbor index to quickly
        find what the closest training data are to the future point).

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        Returns
        -------
        None
        """
        self.train_y = y
        self.NN_index = NearestNeighbors(*self.NN_index_args,
                                         **self.NN_index_kwargs)
        self.NN_index.fit(X)

    def predict_surv(self, X, times, presorted_times=False,
                      limit_from_left=False, n_neighbors=None,
                      kernel_function=None):
        """
        Computes the weighted k-NN Kaplan-Meier survival probability function
        estimate at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the survival
            probability function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to unweighted k-NN
            survival analysis. If a function is specified, then the weighting
            function used is of the form
            "kernel(distance / distance to k-th nearest neighbor)".

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if kernel_function is None:
            kernel_function = lambda s: 1
        dists, indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                                  return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            max_dist = np.max(dist)
            weights = np.array([kernel_function(d / max_dist) for d in dist])
            zero_weight = (weights == 0)
            if zero_weight.sum() > 0:
                weights_subset = weights[~zero_weight]
                if weights_subset.size > 0:
                    output.append(
                        _predict_leaf(
                            _fit_leaf_weighted(train_y[idx[~zero_weight]],
                                               weights_subset),
                            'surv', times, presorted_times, limit_from_left))
                else:
                    output.append(np.ones(n_times))
            else:
                output.append(
                    _predict_leaf(
                        _fit_leaf_weighted(train_y[idx],
                                           weights),
                        'surv', times, presorted_times, limit_from_left))
        return np.array(output)

    def predict_cum_haz(self, X, times, presorted_times=False,
                        limit_from_left=False, n_neighbors=None,
                        kernel_function=None):
        """
        Computes the weighted k-NN Nelson-Aalen cumulative hazard function
        estimate at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the
            cumulative hazard function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to unweighted k-NN
            survival analysis. If a function is specified, then the weighting
            function used is of the form
            "kernel(distance / distance to k-th nearest neighbor)".

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if kernel_function is None:
            kernel_function = lambda s: 1
        dists, indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                                  return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            max_dist = np.max(dist)
            weights = np.array([kernel_function(d / max_dist) for d in dist])
            zero_weight = (weights == 0)
            if zero_weight.sum() > 0:
                weights_subset = weights[~zero_weight]
                if weights_subset.size > 0:
                    output.append(
                        _predict_leaf(
                            _fit_leaf_weighted(train_y[idx[~zero_weight]],
                                               weights_subset),
                            'cum_haz', times, presorted_times, limit_from_left))
                else:
                    output.append(np.zeros(n_times))
            else:
                output.append(
                    _predict_leaf(
                        _fit_leaf_weighted(train_y[idx],
                                           weights),
                        'cum_haz', times, presorted_times, limit_from_left))
        return np.array(output)


def _fit_leaf_weighted(y, weights):
    """
    Computes leaf node information given survival labels (observed times and
    event indicators) that have weights. This is for computing kernel variants
    of the Kaplan-Meier and Nelson-Aalen estimators.

    Parameters
    ----------
    y : 2D numpy array, shape=[n_samples, 2]
        The two columns correspond to observed times and event indicators.

    weights : 1D numpy array, shape=[n_samples]
        Nonnegative weights; i-th weight corresponds to the i-th row in `y`.

    Returns
    -------
    tree : dictionary
        The leaf node information stored as a dictionary. Specifically, the
        key-value pairs of this dictionary are as follows:
        - 'times': stores the sorted unique observed times
        - 'event_counts': in the same order as `times`, the number of events
            at each unique observed time
        - 'at_risk_counts': in the same order as `times`, the number of
            subjects at risk at each unique observed time
        - 'surv': in the same order as `times`, the Kaplan-Meier survival
            probability estimate at each unique observed time
        - 'cum_haz': in the same order as `times`, the Nelson-Aalen cumulative
            hazard estimate at each unique observed time
    """
    if y.size == 0:
        return {'times': sorted_unique_observed_times,
                'event_counts': event_counts,
                'at_risk_counts': at_risk_counts,
                'surv': surv_func,
                'cum_haz': cum_haz_func}

    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    sorted_unique_observed_times = np.sort(np.unique(y[:, 0]))
    num_unique_observed_times = len(sorted_unique_observed_times)
    time_to_idx = {time: idx
                   for idx, time in enumerate(sorted_unique_observed_times)}
    event_counts = np.zeros(num_unique_observed_times)
    dropout_counts = np.zeros(num_unique_observed_times)
    at_risk_counts = np.zeros(num_unique_observed_times)
    at_risk_counts[0] = np.sum(weights)

    for (observed_time, event_ind), weight in zip(y, weights):
        idx = time_to_idx[observed_time]
        if event_ind:
            event_counts[idx] += weight
        dropout_counts[idx] += weight

    for idx in range(num_unique_observed_times - 1):
        at_risk_counts[idx + 1] = at_risk_counts[idx] - dropout_counts[idx]

    surv_prob = 1.
    cum_haz = 0.
    surv_func = np.zeros(num_unique_observed_times)
    cum_haz_func = np.zeros(num_unique_observed_times)
    for idx in range(num_unique_observed_times):
        frac = event_counts[idx] / at_risk_counts[idx]
        surv_prob *= 1 - frac
        cum_haz += frac
        surv_func[idx] = surv_prob
        cum_haz_func[idx] = cum_haz

    return {'times': sorted_unique_observed_times,
            'event_counts': event_counts,
            'at_risk_counts': at_risk_counts,
            'surv': surv_func,
            'cum_haz': cum_haz_func}


def _predict_leaf(tree, mode, times, presorted_times, limit_from_left=False):
    """
    Computes either the Kaplan-Meier survival function estimate or the
    Nelson-Aalen cumulative hazard function estimate at user-specified times
    using survival label data in a leaf node.

    Parameters
    ----------
    tree : dictionary
        Leaf node of a decision tree where we pull survival label information
        from.

    mode : string
        Either 'surv' for survival probabilities or 'cum_haz' for cumulative
        hazard function.

    times : 1D numpy array
        Times to compute the survival probability or cumulative hazard function
        at.

    presorted_times : boolean
        Flag for whether `times` is already sorted.

    limit_from_left : boolean, optional (default=False)
        Flag for whether to output the function evaluated at a time just to the
        left, i.e., instead of outputting f(t) where f is either the survival
        probability or cumulative hazard function estimate, output:
            f(t-) := limit as t' approaches t from the left of f(t').

    Returns
    -------
    output : 1D numpy array
        Survival probability or cumulative hazard function evaluated at each of
        the times specified in `times`.
    """
    unique_observed_times = tree['times']
    surv_func = tree[mode]

    if presorted_times:
        sort_indices = range(len(times))
    else:
        sort_indices = np.argsort(times)

    num_leaf_times = len(unique_observed_times)
    leaf_time_idx = 0
    last_seen_surv_prob = 1.
    output = np.zeros(len(times))
    if limit_from_left:
        for sort_idx in sort_indices:
            time = times[sort_idx]
            while leaf_time_idx < num_leaf_times:
                if unique_observed_times[leaf_time_idx] <= time:
                    last_seen_surv_prob = surv_func[leaf_time_idx]
                    leaf_time_idx += 1
                else:
                    break
            output[sort_idx] = last_seen_surv_prob
        # return np.interp(times, unique_observed_times[1:], surv_func[:-1])
    else:
        for sort_idx in sort_indices:
            time = times[sort_idx]
            while leaf_time_idx < num_leaf_times:
                if unique_observed_times[leaf_time_idx] < time:
                    last_seen_surv_prob = surv_func[leaf_time_idx]
                    leaf_time_idx += 1
                else:
                    break
            output[sort_idx] = last_seen_surv_prob
        # return np.interp(times, unique_observed_times, surv_func)
    return output


class CDFRegressionKNNWeightedSurvival():
    def __init__(self, *args, **kwargs):
        """
        Arguments are the same as for `sklearn.neighbors.NearestNeighbors`.
        The simplest usage of this class is to use a single argument, which is
        `n_neighbors` for the number of nearest neighbors (Euclidean distance
        is assumed in this case). If you want to parallelize across different
        search queries, use the `n_jobs` keyword parameter (-1 to use all
        cores). To use other distances and for other details, please refer to
        the documentation for sklearn's `NearestNeighbors` class.

        *Important:* The prediction methods for this class use weighted
        k-nearest neighbors, where "k" is set equal to the `n_neighbors`
        parameter. The weights are specified through a kernel function K. In
        particular, the i-th nearest neighbor X_i for a test point x is given a
        weight of:
            K( (distance between x and X_i) / (distance between x and X_k) ).
        """
        self.NN_index_args = args
        self.NN_index_kwargs = kwargs
        self.NN_index = None

    def fit(self, X, y):
        """
        Constructs a nearest-neighbor index given training data (so that for
        a future data point, we can use the nearest-neighbor index to quickly
        find what the closest training data are to the future point).

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        Returns
        -------
        None
        """
        self.train_y = y
        self.NN_index = NearestNeighbors(*self.NN_index_args,
                                         **self.NN_index_kwargs)
        self.NN_index.fit(X)

    def predict_surv(self, X, times, presorted_times=False,
                      limit_from_left=False, n_neighbors=None,
                      kernel_function=None):
        """
        Computes the weighted k-NN CDF estimation followed by k-NN regression
        survival probability function estimate at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the survival
            probability function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to unweighted k-NN
            survival analysis. If a function is specified, then the weighting
            function used is of the form
            "kernel(distance / distance to k-th nearest neighbor)".

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if kernel_function is None:
            kernel_function = lambda s: 1
        dists, indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                                  return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            max_dist = np.max(dist)
            weights = np.array([kernel_function(d / max_dist) for d in dist])
            zero_weight = (weights == 0)
            if zero_weight.sum() > 0:
                weights_subset = weights[~zero_weight]
                if weights_subset.size > 0:
                    labels_subset = train_y[idx[~zero_weight]]
                else:
                    output.append(np.ones(n_times))
                    continue
            else:
                labels_subset = train_y[idx]
                weights_subset = weights

            # step 1
            weighted_edf_times, weighted_edf = \
                compute_weighted_edf(labels_subset[:, 0], weights_subset)
            one_minus_weighted_edf = 1 - weighted_edf
            if weighted_edf[0] < 1 and weighted_edf_times[0] > 0:
                weighted_edf_times = \
                    np.concatenate(([0.], weighted_edf_times))
                weighted_edf = \
                    np.concatenate(([1.], weighted_edf))

            # step 2
            denoms = np.interp(labels_subset[:, 0], weighted_edf_times,
                               weighted_edf)
            neg_log_S_est = np.zeros(len(times))
            for time_idx, t in enumerate(times):
                neg_log_S_est[time_idx] = \
                    np.inner(labels_subset[:, 1]
                             * (labels_subset[:, 0] <= t) / denoms,
                             weights_subset)

            output.append(np.exp(-neg_log_S_est))
        return np.array(output)

    def predict_cum_haz(self, X, times, presorted_times=False,
                        limit_from_left=False, n_neighbors=None,
                        kernel_function=None):
        """
        Computes the weighted k-NN CDF estimation followed by k-NN regression
        cumulative hazard function estimate at user-specified times.

        *Important:* The default number of nearest neighbors to use is whatever
        was specified in `args` or `kwargs` when creating an instance of this
        class (the "k" in k-NN)!

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the cumulative hazard function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        limit_from_left : boolean, optional (default=False)
            Flag for whether to output the function evaluated at a time just to
            the left, i.e., instead of outputting f(t) where f is the
            cumulative hazard function estimate, output:
                f(t-) := limit as t' approaches t from the left of f(t').

        n_neighbors : int, None, optional (default=None)
            Number of nearest neighbors to use. If set to None then the number
            used is whatever was passed into `args` or `kwargs` when creating
            an instance of this class.

        kernel_function : function, None, optional (default=None)
            Kernel function to use. None corresponds to unweighted k-NN
            survival analysis. If a function is specified, then the weighting
            function used is of the form
            "kernel(distance / distance to k-th nearest neighbor)".

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        if kernel_function is None:
            kernel_function = lambda s: 1
        dists, indices = self.NN_index.kneighbors(X, n_neighbors=n_neighbors,
                                                  return_distance=True)
        train_y = self.train_y
        output = []
        n_times = len(times)
        for dist, idx in zip(dists, indices):
            max_dist = np.max(dist)
            weights = np.array([kernel_function(d / max_dist) for d in dist])
            zero_weight = (weights == 0)
            if zero_weight.sum() > 0:
                weights_subset = weights[~zero_weight]
                if weights_subset.size > 0:
                    labels_subset = train_y[idx[~zero_weight]]
                else:
                    output.append(np.ones(n_times))
                    continue
            else:
                labels_subset = train_y[idx]
                weights_subset = weights

            # step 1
            weighted_edf_times, weighted_edf = \
                compute_weighted_edf(labels_subset[:, 0], weights_subset)
            one_minus_weighted_edf = 1 - weighted_edf
            if weighted_edf[0] < 1 and weighted_edf_times[0] > 0:
                weighted_edf_times = \
                    np.concatenate(([0.], weighted_edf_times))
                weighted_edf = \
                    np.concatenate(([1.], weighted_edf))

            # step 2
            denoms = np.interp(labels_subset[:, 0], weighted_edf_times,
                               weighted_edf)
            neg_log_S_est = np.zeros(len(times))
            for time_idx, t in enumerate(times):
                neg_log_S_est[time_idx] = \
                    np.inner(labels_subset[:, 1]
                             * (labels_subset[:, 0] <= t) / denoms,
                             weights_subset)

            output.append(neg_log_S_est)
        return np.array(output)


def compute_weighted_edf(obs, weights=None):
    """
    Computes a weighted empirical distribution function.

    Parameters
    ----------
    obs : 1D numpy array
        Observations to construct the weighted empirical distribution from.

    weights : 1D numpy array, None, optional (default=None)
        Nonnegative weights for the observations. The i-th weight corresponds
        to the i-th value in `obs`. None refers to using uniform weights,
        i.e., each point has weight 1/len(obs).

    Returns
    -------
    sorted_unique_obs : 1D numpy array
        Sorted unique observations in ascending order.

    weighted_edf : 1D numpy array
        The weighted empirical distribution function evaluated at each of the
        values in `sorted_unique_obs`, in the same order.
    """
    if weights is None:
        weights = np.ones(len(obs))
        weights /= weights.shape[0]

    sorted_unique_obs = np.sort(np.unique(obs))
    obs_to_idx = {obs: idx for idx, obs in enumerate(sorted_unique_obs)}
    weighted_edf = np.zeros(len(sorted_unique_obs))
    for x, w in zip(obs, weights):
        weighted_edf[obs_to_idx[x]] += w

    weighted_edf = np.cumsum(weighted_edf)
    return sorted_unique_obs, weighted_edf

def _fit_leaf(y):
    """
    Computes leaf node information given survival labels (observed times and
    event indicators).

    Parameters
    ----------
    y : 2D numpy array, shape=[n_samples, 2]
        The two columns correspond to observed times and event indicators.

    Returns
    -------
    tree : dictionary
        The leaf node information stored as a dictionary. Specifically, the
        key-value pairs of this dictionary are as follows:
        - 'times': stores the sorted unique observed times
        - 'event_counts': in the same order as `times`, the number of events
            at each unique observed time
        - 'at_risk_counts': in the same order as `times`, the number of
            subjects at risk at each unique observed time
        - 'surv': in the same order as `times`, the Kaplan-Meier survival
            probability estimate at each unique observed time
        - 'cum_haz': in the same order as `times`, the Nelson-Aalen cumulative
            hazard estimate at each unique observed time
    """
    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    sorted_unique_observed_times = np.sort(np.unique(y[:, 0]))
    num_unique_observed_times = len(sorted_unique_observed_times)
    time_to_idx = {time: idx
                   for idx, time in enumerate(sorted_unique_observed_times)}
    event_counts = np.zeros(num_unique_observed_times)
    dropout_counts = np.zeros(num_unique_observed_times)
    at_risk_counts = np.zeros(num_unique_observed_times)
    at_risk_counts[0] = len(y)

    for observed_time, event_ind in y:
        idx = time_to_idx[observed_time]
        if event_ind:
            event_counts[idx] += 1
        dropout_counts[idx] += 1

    for idx in range(num_unique_observed_times - 1):
        at_risk_counts[idx + 1] = at_risk_counts[idx] - dropout_counts[idx]

    surv_prob = 1.
    cum_haz = 0.
    surv_func = np.zeros(num_unique_observed_times)
    cum_haz_func = np.zeros(num_unique_observed_times)
    for idx in range(num_unique_observed_times):
        frac = event_counts[idx] / at_risk_counts[idx]
        surv_prob *= 1 - frac
        cum_haz += frac
        surv_func[idx] = surv_prob
        cum_haz_func[idx] = cum_haz

    return {'times': sorted_unique_observed_times,
            'event_counts': event_counts,
            'at_risk_counts': at_risk_counts,
            'surv': surv_func,
            'cum_haz': cum_haz_func}
