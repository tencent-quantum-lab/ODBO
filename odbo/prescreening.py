import numpy as np
import warnings
from pyod.models.xgbod import XGBOD as xgbod


def sp_label(X, Y, thres=None, fraction=0.1):
    """Get inlier (0) and outlier (1) for given X, y values
    Parameters
    ----------
    X : ndarray with a shape of (n_training_samples, feature_size)
        Current set of experiments.
    Y : ndarray with a shape of (n_training_samples, 1) of floats
        Current measurements using X experimental design. 
    thres : float, default=None
        Threshold used to select the inliers and outiers
    fraction : float, default=0.1
        If there is no threshold provided, the fraction of experiments can be 
        outliers in the current experiments
    Returns
    -------
    labels : ndarray of zero or one
        Final vector for experiements marked as inliers and outliers
    """

    labels = np.zeros(len(Y))
    if thres == None:
        sort_ids = np.argsort(Y)
        labels[sort_ids[0:int(len(Y) * fraction)]] = np.ones(
            int(len(Y) * fraction))
    else:
        outlier = [k for k, x in enumerate(Y) if x <= thres]
        labels[outlier] = np.ones(len(outlier))
    return labels


class XGBOD(xgbod):
    """XGBOD class (pass to pyOD package)"""

    def __init__(self, *args, **kwargs):
        super(XGBOD, self).__init__(*args, **kwargs)
        if 'estimator_list' not in kwargs:
            from pyod.models.knn import KNN
            from pyod.models.lof import LOF
            from pyod.models.ocsvm import OCSVM
            from pyod.models.iforest import IForest
            self.estimator_list = [KNN(), LOF(), OCSVM(), IForest()]


class LAMCTS():
    pass
