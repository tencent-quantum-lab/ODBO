import numpy as np
import torch


def normalize_data(X, Y, X_pending=None):
    """Util function to normalize the input data to [0,1]^d for TuRBO algorithm
    Parameters
    ----------
    X : pyTorch tensor with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation. 
    Y : pyTorch tensor with a shape of (n_training_samples, 1) of floats
        Current measurements using X experimental design. 
    X_pending : pyTorch tensor with a shape of (n_pending_samples, feature_size) of floats
        Current search space of experimental design after featurziation. 
    Returns
    -------
    train_x : Normalized tensor of training X
    train_y : Normalized tensor of training y 
    test_x :  Normalized tensor of search space (pending X)
    stats : Statstical values of the normalization [Mean, Std, Min, Max]  of each feature in X
    """
    train_y = (Y - torch.mean(Y)) / torch.std(Y)
    if X_pending != None:
        X_combine = torch.cat([X, X_pending])
        X_mean, X_std = torch.mean(
            X_combine, dim=0), torch.std(
                X_combine, dim=0)
        X_combine = (X_combine - X_mean) / X_std
        X_min, X_max = torch.min(
            X_combine, dim=0)[0], torch.max(
                X_combine, dim=0)[0]
        train_x, test_x = (X - X_mean) / X_std, (X_pending - X_mean) / X_std
        train_x, test_x = torch.div(train_x - X_min, X_max - X_min), torch.div(
            test_x - X_min, X_max - X_min)
    else:
        X_mean, X_std = torch.mean(X, dim=0), torch.std(X, dim=0)
        train_x = (X - X_mean) / X_std
        X_min, X_max = torch.min(train_x, dim=0)[0], torch.max(train_x, dim=0)[0]
        train_x = torch.div(train_x - X_min, X_max - X_min)
        test_x = None
    stats = [X_mean, X_std, X_min, X_max]
    return train_x, train_y, test_x, stats


def denormalize_X(train_x, stats):
    """Util function to recover the orginal X given the normalization stats 
    Parameters
    ----------
    train_x : Normalized pyTorch tensor with a shape of (n_training_samples, feature_size) of 
        values in [0,1]. Current set of experimental design after featurziation. 
    stats : Statstical values of the normalization [Mean, Std, Min, Max]  of each feature in X
    Returns
    -------
    X : pyTorch tensor with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation. 
    """

    train_x = torch.multiply(stats[3] - stats[2], train_x) + stats[2]
    X = train_x * stats[1] + stats[0]
    return X


def code_to_array(X):
    """Util function to make a sequence string to a list of codes 
    Parameters
    ----------
    X : List of n raw experiments expressed in a sequence
    Returns
    -------
    name : nd.array of codes with shape of (n, length of the sequence)
    """
    name = []
    for i in range(len(X)):
        name.append(list(X[i]))
    name = np.vstack(name)
    return name
