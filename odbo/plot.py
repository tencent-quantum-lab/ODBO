import matplotlib.pyplot as plt
import numpy as np
import warnings


def plot_cm(true_labels, pred_labels, Y=None, cmap=None):
    """Plot the confusion matrix using heat map
    Parameters
    ----------
    true_labels : list or array of zero or ones
        True inliers (zeros) and outliers (ones)
    pred_labels :  list or array of zero or ones
        Predicted inliers (zeros) and outliers (ones) by prescreening model
    Y : list or array of floats
        All measurements values
    cmap : colormap code
        Color map code used in matplotlib 
    Returns
    -------
    out_outlier : list of ints
        Indices of experiments that is outlier in both true values and predictions
    in_outlier : list of ints
        Indices of experiments that is outlier in true values but inlier in predictions
    out_inlier : list of ints 
        Indices of experiments that is inlier in true values but outlier in predictions 
    in_inlier : list of ints 
        Indices of experiments that is inlier in both true and predictions 
    """
    outlier = np.array([k for k, x in enumerate(pred_labels) if x == 1])
    inlier = np.array([k for k, x in enumerate(pred_labels) if x == 0])
    out_outlier = outlier[[
        k for k, x in enumerate(true_labels[outlier]) if x == 1
    ]]
    in_outlier = outlier[[
        k for k, x in enumerate(true_labels[outlier]) if x == 0
    ]]
    out_inlier = inlier[[
        k for k, x in enumerate(true_labels[inlier]) if x == 1
    ]]
    in_inlier = inlier[[
        k for k, x in enumerate(true_labels[inlier]) if x == 0
    ]]
    true = ['True Outlier', 'True Inlier']
    pred = ['Pred Outlier', 'Pred Inlier']
    if Y is None:
        avg = np.ones((2, 2)) * np.inf
    else:
        coo, cio, coi, cii = np.mean(Y[out_outlier]), np.mean(
            Y[in_outlier]), np.mean(Y[out_inlier]), np.mean(Y[in_inlier])
        avg = np.array([[coo, cio], [coi, cii]])

    count = np.array([[len(out_outlier), len(in_outlier)],
                      [len(out_inlier), len(in_inlier)]])
    fig, ax = plt.subplots()
    from matplotlib import cm
    im = ax.imshow(avg, cmap=cm.coolwarm)
    if cmap == None:
        cmap = "YlGn"
    ax.set_xticks(np.arange(len(true)))
    ax.set_yticks(np.arange(len(pred)))
    ax.set_xticklabels(true)
    ax.set_yticklabels(pred)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(pred)):
        for j in range(len(true)):
            text = ax.text(
                j,
                i,
                str(round(avg[i, j], 4)) + '\nRatio {0:.3%}'.format(
                    count[i, j] / len(pred_labels), 4),
                ha="center",
                va="center",
                color="k")
    if avg.all() != np.inf:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Measurments', rotation=-90, va="bottom")
    ax.set_title("Confusion matrix plot for this experiment")
    fig.tight_layout()
    return out_outlier, in_outlier, out_inlier, in_inlier


def plot_hist(X, Y=None, mode='count'):
    """Plot the histogram of experiments with measurments
    Parameters
    ----------
    X : ndarray (n_training_samples, feature_size) of floats
        Current set of experimental designs.
    Y : list or array of floats
        All measurements values
    mode : str 
        Plotting mode. Must be 'count' or 'measurements'
    """
    if Y is None and mode == 'measurement':
        warnings.warn(
            "'measurement' plotting mode require to have experimental measurement values Y"
        )
        mode = 'count'
    fig, axs = plt.subplots(
        1, X.shape[1], figsize=(8 * X.shape[1], 8), sharey=True)
    for i in range(X.shape[1]):
        if mode == 'count':
            values = np.empty((len(categories), 2))
        elif mode == 'measurements':
            values = np.empty((len(categories), 3))

        categories = list(set(X[:i]))
        for j in range(len(categories)):
            ids_vec = np.array(
                [k for k, x in enumerate(X[:i]) if x == categories[j]])
            values[j, 0] = categories[j]
            if mode == 'count':
                values[j, 1] = len(ids_vec)
            elif mode == 'measurement':
                values[j, 1] = np.mean(Y[ids_vec])
                values[j, 2] = np.std(Y[ids_vec])
        axs[i].bar(values[:, 0], values[:, 1])
        if mode == 'measurement':
            axs[i].errorbar(
                values[:, 0],
                values[:, 1],
                yerr=values[:, 2],
                fmt="o",
                color="r")
            if i == 0:
                axs[i].set_ylabel('Measurements', fontsize=18)
        elif mode == 'count':
            if i == 0:
                axs[i].set_ylabel('Counts', fontsize=18)
        axs[i].set_title('Variable %d' % k, fontsize=24)
        axs[i].set_xlabel('Categories', fontsize=24)
        axs[i].set_xticklabels(labels=values[:, 0], fontsize=18)


def plot_bo(Y, Y_std, methods=None, cmap=None):
    pass
