import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sksurv.ensemble import RandomSurvivalForest
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.tree import SurvivalTree
from .indexcalculator import IndexCalculator
from cobsurv.utils import predict_probability_at_times
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from .npsurvival_models import KNNSurvival


class CobraSurvival(BaseEstimator):
    """
    This is COBRA implementation for survival analysis, it is a meta-learner
    that combines the predictions of several survival cobsurv to produce a
    final prediction. The algorithm is described in the paper:



    .. math::


        \\hat{S}_{COBRA}(t | x; \\cdot ) = \\prod_{t' \\in \\mathcal{Y}_{\\Gamma}(x; \\cdot)}
         \\left( 1 - \\frac{\\mathcal{D}_{\\Gamma}(t' | x; \\cdot )}{\mathcal{R}_{\\Gamma}(t' | x; \\cdot)} \\right)^{\\mathbb{I}(t' \\leq t)}




    See the :ref:`User Guide </user_guide/cobrasurvival.ipynb>`, and the research paper
    [1]_  for further description.

    Parameters
    ----------
    epsilon: float , default is 0.12
        The :math:`\\epsilon` proximity which checks if the two machines survival prediction are
        close :math:`d(s_1,s_2) < \\epsilon`

    alpha: integer , default is 4
        The :math:`\\alpha` the number of machines which are in consensus of :math:`\\epsilon` proximity

    machines: list of scikit-learn type of machines, default is "default"
        machines, parameter, set to default for now, The defaults are Random Survival Forest, Cox Ridge,
        Cox Lasso, Survival Tree, KNN Survival but user can provide their own machines with conditions that it have a
        predict_survival_function method which returns a numpy array of shape (n_samples,n_times), the survival
        prediction at times of unique times of training data.

    distance_function: string or callable, default is "euler"
        distance function, parameter, set to default for now, The default is the area type
        of norm, as used in the paper, but user can provide their own distance function which takes two numpy
        arrays of shape (n_times,) and returns a float. The distance function can be "trapz", "euclidean",
        "cross_entropy", "cross_entropy_from_mean", "shannon_jensen", "kl", "max", "euler" or a callable function.


    Attributes
    ----------
    unique_sorted_times_l: 1d array
        Unique sorted time points of happening of an event or censoring in dataset :math:`D_l`.
    unique_sorted_times_k: 1d array
        Unique sorted time points of happening of an event or censoring in dataset :math:`D_k`.
    pred_l : 3d array
        The survival prediction of machines on dataset :math:`D_l` of shape (n_samples,n_estimators,n_times).
    epsilon : float
        The :math:`\\epsilon` proximity which checks if the two machines survival prediction sre close :math:`d(s_1,s_2) < \\epsilon`.
    alpha : integer
        The :math:`\\alpha` the number of machines which are in consensus of $\\epsilon$ proximity.
    n_estimators : int , default is 5
        The number of machines.
    indexer : A IndexCalculator object
        The IndexCalculator object which is used to calculate the index.
    train_time : 1d array
        The  time points of dataset :math:`D_l`.
    train_event : 1d array
        A boolean indicator for censoring True represents the happening of an event of dataset :math:`D_l`.



    See also
    --------
    cobsurv.distance_functions : The distance functions used in COBRA
        A single survival tree.

    Notes
    -----
    The COBRA algorithm is described in the paper [1]_. We must know that as we increase :math:`\\epsilon` the
    survival curve will become more like the population survival curve.


    References
    ----------
    .. [1] Rahul Goswami, and Arabin K. Dey. "Area-norm COBRA on Conditional Survival Prediction."
        ArXiv, (2023). Accessed September 6, 2023. https://arxiv.org/abs/2309.00417.


    """

    def __init__(self, epsilon=0.12, alpha=4, machines="default", distance_function="euler"):

        self.epsilon = epsilon
        self.alpha = alpha
        self.machines = machines
        self.distance_function = distance_function

    def fit(self, X, y, n_quantiles=10, l_by_n=0.5, experimental=False):
        """
        This part does the splitting the dataset in two parts $X_l$ and $X_k$ and $y_l$ and $y_k$,
        the splitting is done strategically to ensure that censoring and event times are distributed
        equally in both the splits, Secondly this function train the initial cobsurv on $D_k$ dataset



        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : structured numpy array of shape (n_samples,)
            The target variable with the first field as the event indicator and the second field as the event time
        n_quantiles : int, default=10
            The number of quantiles to be used for splitting the data, defult is set to 10
        l_by_n : float, default=0.5
            The split ratio, defult is set to 0.5
        experimental : bool, default=False
            If set to True the cobsurv are trained on whole dataset and no splitting will happen , default is False

        Returns
        -------
        self : object

        """



        if not isinstance(n_quantiles, int) or n_quantiles <= 1:
            raise ValueError("The number of quantiles must be an integer greater than 1.")

        # Ensure that the split ratio is within the valid range of 0 to 1
        if not (0 < l_by_n < 1):
            raise ValueError("The split ratio must be a value between 0 and 1 (exclusive).")

        # Ensure that the input data 'X' is a numpy array
        if not isinstance(X, np.ndarray) and X.ndim != 2:
            raise TypeError("Input 'X' must be a numpy array.")

        # Ensure that the target variable 'y' is a numpy array
        if not isinstance(X, np.ndarray):
            raise TypeError("Target variable 'y' must be a numpy array.")

        assert X.shape[0] == y.shape[
            0], "The number of samples in 'X' and 'y' must be equal, but got {0} and {1} respectively.".format(
            X.shape[0], y.shape[0])

        _, self.n_features_in_ = X.shape

        while True:
            try:
                time = y[y.dtype.names[1]]
                strat_array = pd.qcut(time, q=n_quantiles, labels=False, duplicates='drop')
                result = [str(x) + str(y) for x, y in zip(strat_array, (y[y.dtype.names[0]]))]
                Xl, Xk, yl, yk = train_test_split(X, y, test_size=l_by_n, stratify=result, random_state=2)
                break
            except:
                print("Warning : The number of quantiles is too high, reducing the number of quantiles by 1")
                n_quantiles = n_quantiles - 1

        if experimental:
            Xl, Xk = X, X
            yl, yk = y, y

        self.Xl = Xl  # needed for covariate relevance calculation

        self.train_time = yl[yl.dtype.names[1]]
        self.train_event = yl[yl.dtype.names[0]].astype(bool)
        self.unique_sorted_times_l = np.sort(np.unique(yl[yl.dtype.names[1]])).astype(float)
        self.unique_sorted_times_k = np.sort(np.unique(yk[yk.dtype.names[1]])).astype(float)
        self._fit_initial_machines(Xk, yk)
        self.pred_l = self._get_initial_predictions(Xl)

        return self

    def _fit_initial_machines(self, Xk, yk):
        """
        Fit the initial machines on the dataset $D_k$

        Parameters
        ----------
        Xk : array-like of shape (n_samples, n_features)
            The input samples.
        yk : structured numpy array of shape (n_samples,)
            The target variable with the first field as the event indicator and the second field as the event time
        """
        if self.machines == "default":
            self.rf = RandomSurvivalForest(n_estimators=100, random_state=42)
            self.rf.fit(Xk, yk)

            self.knn = KNNSurvival()
            self.knn.fit(Xk, np.array([yk[yk.dtype.names[1]], yk[yk.dtype.names[0]]]).T)

            self.coxridge = CoxnetSurvivalAnalysis(l1_ratio=0.0000001, fit_baseline_model=True)
            self.coxridge.fit(Xk, yk)

            self.coxlasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True)
            self.coxlasso.fit(Xk, yk)

            self.tree = SurvivalTree(random_state=42)
            self.tree.fit(Xk, yk)
        else:
            if isinstance(self.machines, list):
                for machine in self.machines:
                    assert callable(machine.fit) and callable(machine.predict_survival_function), \
                        f"The machine must have fit and predict_survival_function method , but for {machine} it is not the case"
                    assert hasattr(machine, "unique_times_"), \
                        f"The machine must have unique_times_ attribute , but for {machine} it is not the case"
                self.machines = self.machines
            else:
                raise TypeError(
                    "The machines must be a list of machines , which have fit and predict_survival_function method")
            fitted_machines = []
            for machine in self.machines:
                try:
                    machine.fit(Xk, yk)
                    fitted_machines.append(machine)
                except:
                    print(f"Warning : The machine {machine} is not fitted on the dataset")
            self.machines = fitted_machines

    def _get_initial_predictions(self, X):
        """
        Get the initial predictions from the machines
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        """
        if self.machines == "default":
            self.rf_pred = self.rf.predict_survival_function(X, return_array=True)
            self.knn_pred = self.knn.predict_surv(X, self.unique_sorted_times_k)
            self.tree_pred = self.tree.predict_survival_function(X, return_array=True)
            self.ridge_pred = self.coxridge.predict_survival_function(X, return_array=True)
            self.lasso_pred = self.coxlasso.predict_survival_function(X, return_array=True)
            self.n_estimators = 5
            result = np.array([self.rf_pred, self.knn_pred, self.tree_pred, self.lasso_pred, self.ridge_pred])
        else:
            pred_list = []
            for machine in self.machines:
                try:
                    pred = machine.predict_survival_function(X)
                    pred = predict_probability_at_times(machine.unique_times_, pred, self.unique_sorted_times_k)
                    pred_list.append(pred)
                except:
                    print(f"Warning : The machine {machine} is not fitted on the dataset")

            result = np.array(pred_list)

            self.n_estimators = len(pred_list)
            del pred_list

        return np.transpose(result, (1, 0, 2)).astype(np.float32)

    def predict(self, X):
        """
        Predict the survival function for a given set of covariates

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        """

        query_pred = self._get_initial_predictions(X)

        self.indexer = IndexCalculator(query_pred, self.pred_l, self.unique_sorted_times_k)
        index_arr = self.indexer.get_index(self.epsilon, self.alpha, self.distance_function)

        return self._get_survival_function(index_arr)

    def _get_survival_function(self, index_arr):
        """
        Get the survival function from the selected machines

        Parameters
        ----------
        index_arr : array-like of shape (n_samples, n_samples in $D_l$)
            The index array which is used to select the proximity observations from :math:`D_l`
        """
        estimate = np.zeros((index_arr.shape[0], len(self.unique_sorted_times_l)))
        for i in range(index_arr.shape[0]):
            time, prob = kaplan_meier_estimator(self.train_event[index_arr[i]], self.train_time[index_arr[i]])
            estimate[i] = predict_probability_at_times(time, prob, self.unique_sorted_times_l)
        return estimate

    def get_covariate_relevance(self, X=None):
        """
        Get the covariate relevance of the covariates
        This function computes the relevance with respect to given covariates, if no covariates are given then it
        computes the relevance with respect to average of all the in the set $D_l$

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        """
        if X is None:
            query_x = self.Xl.mean(axis=0)
            query_pred = self._get_initial_predictions(query_x.reshape(1, -1))
        else:
            query_pred = self._get_initial_predictions(X.reshape(1, -1))
        index_arr = IndexCalculator(query_pred, self.pred_l, self.unique_sorted_times_k).get_index(self.epsilon,
                                                                                                   self.alpha,
                                                                                                   self.distance_function)
        model = LogisticRegression(fit_intercept=False)
        model.fit(self.Xl, index_arr)
        return model.coef_

    def _get_ev(self, X, y):
        """
        The package uses EvalSurv package from pycox to calculate the concordance and integrated brier score, and various
        other metrics that are available in the package, also it uses ploting of pycox to plot the survival function

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : structured numpy array of shape (n_samples,)
            The target variable with the first field as the event indicator and the second field as the event time
        """

        prediction = self.predict(X)
        prediction = pd.DataFrame(prediction.T, index=self.unique_sorted_times_l)
        ev = EvalSurv(prediction, y[y.dtype.names[1]], y[y.dtype.names[0]], censor_surv='km')
        return ev

    def score(self, X, y, type="ibs"):
        """
        The score function provides the integrated brier score by default, if type is set to concordance then it provides
        the concordance index

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : structured numpy array of shape (n_samples,)
            The target variable with the first field as the event indicator and the second field as the event time
        type : string, default is "ibs"
            The type of score to be calculated, it can be "ibs" or "concordance"

        Returns
        -------
        score : float
            The score of the model
        """
        ev = self._get_ev(X, y)
        if type == "ibs":
            return ev.integrated_brier_score(self.unique_sorted_times_l)
        elif type == "concordance":
            return ev.concordance_td()

    def plot(self, X, index):
        """
        This plot the survival function of the given covariates

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        index : integer
            The index of the covariate to be plotted
        """
        event = np.random.random(X.shape[0]) < 0.5
        time = np.random.randint(0, 365, size=X.shape[0])
        y = np.array(list(zip(event, time)), dtype=[('event', '?'), ('time', '<f8')])
        ev = self._get_ev(X, y)
        ev[index].plot_surv()



