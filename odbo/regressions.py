"""Regression algorithms"""

import numpy as np
import torch
import gpytorch
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch, fit_gpytorch_scipy
from .gp import StudentTGP, GP


def GPRegression(X,
                 Y,
                 likelihood=None,
                 noise_constraint=gpytorch.constraints.Interval(1e-6, 1e-2),
                 min_inferred_noise_level=1e-4,
                 optimizer='fit_gpytorch_scipy',
                 **kwargs):
    """Surrograte model using exact single task Gaussian process regression.
    Parameters
    ----------
    X : pyTorch tensor with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation. 
    Y : pyTorch tensor with a shape of (n_training_samples, 1) of floats
        Current measurements using X experimental design. 
    likelihood : A likelihood. default=None
        Likelihood used in this function.
    noise_constraint : gpytorch.constraints.Interval, default=gpytorch.constraints.
        Interval(1e-6, 1e-2)
        Noise constraints used in the regression.
    min_inferred_noise_level : float, default=1e-4
        Minimum value of added noises to kernel 
    optimizer : str, default='fit_gpytorch_scipy'
        Optimizer used in the regression, must be 'fit_gpytorch_scipy' or 'fit_gpytorch_torch'.
    Returns
    -------
    model : GpyTorch Gaussian process model

    References
    ----------
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, 
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian 
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    """
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.likelihoods import GaussianLikelihood
    from botorch.fit import fit_gpytorch_model
    model = GP(
        X,
        Y,
        likelihood=likelihood,
        min_inferred_noise_level=min_inferred_noise_level,
        **kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if optimizer == 'fit_gpytorch_scipy' or optimizer is None:
        mll.train()
        fit_gpytorch_scipy(mll)
        mll.eval()
    elif optimizer == 'fit_gpytorch_torch':
        mll.train()
        fit_gpytorch_torch(
            mll, options={'maxiter': 500}, track_iterations=False)
        mll.eval()
    else:
        fit_gpytorch_model(mll, optimizier=optimizer)
    return model


def RobustRegression(X,
                     Y,
                     noise_constraint=gpytorch.constraints.Interval(
                         1e-6, 1e-2),
                     min_inferred_noise_level=1e-4,
                     optimizer=None,
                     maxiter=100,
                     thres=0.001,
                     std_factor=2,
                     **kwargs):
    """Surrograte model using exact robust then single task Gaussian process regression.
    Parameters
    ----------
    X : pyTorch tensor with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation. 
    Y : pyTorch tensor with a shape of (n_training_samples, 1) of floats
        Current measurements using X experimental design. 
    noise_constraint : gpytorch.constraints.Interval, default=gpytorch.constraints.
        Interval(1e-6, 1e-2)
        Noise constraints used in the regression.
    min_inferred_noise_level : float, default=1e-4
        Minimum value of added noises to kernel 
    optimizer : str, default='fit_gpytorch_scipy'
        Optimizer used in the regression, must be 'fit_gpytorch_scipy' or 'fit_gpytorch_torch'.
    maxiter : int, default=100
        Maximum optimization iterations.
    thres : float, default=0.001
        Threshold for optimization
    std_factor : int, default=2    
        Outlier dection in the robust regression. Points with prediction +/- std_factor*std 
        are marked as outliers
    Returns
    -------
    model : GpyTorch Gaussian process model

    References
    ----------
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, 
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian 
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    """
    from gpytorch.mlls import VariationalELBO
    from gpytorch.likelihoods import StudentTLikelihood
    likelihood = StudentTLikelihood(noise_constraint=noise_constraint)
    model = StudentTGP(
        X, Y, min_inferred_noise_level=min_inferred_noise_level, **kwargs)
    model.train()
    likelihood.train()
    mll = VariationalELBO(likelihood, model, Y.ravel().numel())
    lossvalues = []
    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for i in range(maxiter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Y.ravel())
        loss.backward()
        lossvalues.append(loss.item())
        if i >= 50 and abs(lossvalues[-1] -
                           np.mean(lossvalues[-10:])) <= thres:
            break
        optimizer.step()
    model.eval()
    with torch.no_grad():
        observed_pred = model(X)
        pred_labels = np.mean(likelihood(observed_pred).mean.numpy(), axis=0)
        std = np.sqrt(observed_pred.variance.numpy())
        mean_std = np.mean(std)
        inlier_ids, outlier_ids = [], []
        for m in range(len(std)):
            if std[m] < std_factor * mean_std:
                inlier_ids.append(m)
            else:
                outlier_ids.append(m)
    return model, inlier_ids, outlier_ids
