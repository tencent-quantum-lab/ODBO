import numpy as np
import random
import botorch
import torch
from gpytorch.utils.errors import NanError, NotPSDError
from .regressions import GPRegression, RobustRegression, HeteroskedasticGPRegression
from .utils import normalize_data
import warnings


def bo_design(X,
              Y,
              X_pending=None,
              gp_method='gp_regression',
              batch_size=1,
              bounds = None,
              min_inferred_noise_level=1e-4,
              acqfn='ei',
              normalize=False,
              verbose=False):
    """Run experimental design using BO
    Parameters
    ----------
    X : pyTorch tensor with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation. 
    Y : pyTorch tensor with a shape of (n_training_samples, 1) of floats
        Current measurements using X experimental design. 
    X_pending : pyTorch tensor with a shape of (n_pending_samples, feature_size) of floats
        Current search space of experimental design after featurziation. 
    gp_method : str, default='gp_regression'
        Regression method used in this run. Must be 'gp_regression' or 'robust_regression'
    batch_size : int, default=1
        Number of next experiments to be added
    min_inferred_noise_level : float, default=1e-4
        Minimum value of added noises to kernel 
    acqfn : str, default='ei'
        Acqusition function used
    verbose : boolean, default=False
        Print out the details of selcted experiments
    Returns
    -------
    X_next : pyTorch tensor with a shape of (batch_size, feature_size) of floats
        Selected experiments by BO.
    acq_value : pyTorch tensor with a shape of of shape (batch_size,) of floats
        Acqusition values for the selected experiments
    next_exp_id : list of indices of length of batch_size
        Indices of selected X with length of batch_size in pending X 
    References
    ----------
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, 
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian 
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    """
    from .bo import generate_batch
    if bounds == None:
        bounds = torch.vstack([torch.zeros(X.shape[1]), torch.ones(X.shape[1])])

    if normalize:
        X_norm, Y_norm, X_pending_norm, stats = normalize_data(
            X, Y, X_pending=X_pending)
    else:
        X_norm, Y_norm, X_pending_norm = X, Y, X_pending

    if gp_method == 'robust_regression':
        model, inliers, outliers = RobustRegression(
            X_norm, Y_norm, min_inferred_noise_level=min_inferred_noise_level)
        X_norm, Y_norm = X_norm[inliers, :], Y_norm[inliers]
        if len(inliers) != len(Y) and verbose == True:
            print(len(Y) - len(inliers), ' outliers found')
        del model, inliers, outliers
    while True:
        try:
            gp_model = GPRegression(
                X_norm,
                Y_norm,
                min_inferred_noise_level=min_inferred_noise_level)
            break
        except:
            gp_model = GPRegression(
                X_norm,
                Y_norm,
                min_inferred_noise_level=min_inferred_noise_level * 10,
                optimizer='fit_gpytorch_torch')
            print(
                'The scipy optimizer and minimum inferred noises cannot make the kernel PSD, switch to torch optimizer'
            )
            break

    X_next, acq_value = generate_batch(model=gp_model,
                                       X=X_norm,
                                       Y=Y_norm,
                                       bounds = bounds,
                                       batch_size=batch_size,
                                       X_pending=X_pending_norm,
                                       acqfn=acqfn)
    next_exp_id = []

    del gp_model, X_norm, Y_norm, X, Y, X_pending
    if X_pending_norm is not None:
        for j in range(batch_size):
            tonext = np.where(
                np.all(X_pending_norm.detach().numpy() == X_next[
                    j:j + 1, :].detach().numpy(),
                       axis=1) == True)[0]
            if len(tonext) > 1:
                tonext = [random.choice(tonext)]
            next_exp_id.extend(tonext)

        if verbose == True:
            print("Next experiment to pick: ",
                  X_next.detach().numpy(), "Acqusition value: ",
                  acq_value.detach().numpy())
    del X_pending_norm

    return X_next, acq_value, next_exp_id


def turbo_design(state,
                 X,
                 Y,
                 Y_var=None,
                 n_trust_regions=1,
                 X_pending=None,
                 gp_method='gp_regression',
                 batch_size=1,
                 min_inferred_noise_level=1e-4,
                 a = 0.2,
                 acqfn='ei',
                 normalize=False,
                 verbose=False):
    """Run experimental design using TuRBO
    Parameters
    ----------
    state : TurboState
        Current state of TuRBO to determine the trust lengths
    X : pyTorch tensor with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation. 
    Y : pyTorch tensor with a shape of (n_training_samples, 1) of floats
        Current measurements using X experimental design. 
    n_trust_regions: int, default=1
        Number of trust regions used in TuRBO. m value in TuRBO-m is the same as this
        n_trust_regions. Default is n_trust_regions=1 (TuRBO-1)
    X_pending : pyTorch tensor with a shape of (n_pending_samples, feature_size) of floats
        Current search space of experimental design after featurziation. 
    gp_method : str, default='gp_regression'
        Regression method used in this run. Must be 'gp_regression' or 'robust_regression'
    batch_size : int, default=1
        Number of next experiments to be added
    min_inferred_noise_level : float, default=1e-4
        Minimum value of added noises to kernel 
    acqfn : str, default='ei'
        Acqusition function used 
    verbose : boolean, default=False
        Print out the details of selcted experiments
    Returns
    -------
    X_next : pyTorch tensor with a shape of (batch_size, feature_size) of floats
        Selected experiments by BO.
    acq_value : pyTorch tensor with a shape of of shape (batch_size,) of floats
        Acqusition values for the selected experiments
    next_exp_id : list of indices of length of batch_size
        Indices of selected X with length of batch_size in pending X 
    References
    ----------
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, 
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian 
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    """
    from .turbo import generate_batch
    if normalize:
        X_norm, Y_norm, X_pending_norm, stats = normalize_data(
            X, Y, X_pending=X_pending)
    else:
        X_norm, Y_norm, X_pending_norm = X, Y, X_pending
    if gp_method == 'robust_regression':
        model, inliers, outliers = RobustRegression(
            X_norm, Y_norm, min_inferred_noise_level=min_inferred_noise_level)
        X_norm, Y_norm = X_norm[inliers, :], Y_norm[inliers]
        if len(inliers) != len(Y) and verbose == True:
            print(len(Y) - len(inliers), ' outliers found')
        del model, inliers, outliers
    if gp_method == 'robust_regression' or gp_method == 'gp_regression':
        while True:
            try:
                gp_model = GPRegression(
                    X_norm,
                    Y_norm,
                    min_inferred_noise_level=min_inferred_noise_level)
                break
            except:
                gp_model = GPRegression(
                    X_norm,
                    Y_norm,
                    min_inferred_noise_level=min_inferred_noise_level * 10,
                    optimizer='fit_gpytorch_torch')
                print(
                    'The scipy optimizer and minimum inferred noises cannot make the kernel PSD, switch to torch optimizer'
                )
                break
    elif gp_method == 'heteroskedastic_regression':
        #stats = [X_mean, X_std, X_min, X_max]
        if normalize:
            Y_var_norm = ((Y_var-stats[0])/stats[1]-stats[2])/(stats[3]-stats[2])
        else:
            Y_var_norm = Y_var
            gp_model = botorch.models.gp_regression.HeteroskedasticSingleTaskGP(X_norm, Y_norm, Y_var_norm)
    if normalize:
        index = np.arange(X_norm.shape[0])
    else:
        index = np.where((np.min(np.array(X_norm), axis=1) >=0) & (np.max(np.array(X_norm), axis=1) <=1))[0]

    X_next, acq_value = generate_batch(state=state,
                                       model=gp_model,
                                       X=X_norm[index, :],
                                       Y=Y_norm[index],
                                       n_trust_regions=n_trust_regions,
                                       batch_size=batch_size,
                                       X_pending=X_pending_norm,
                                       a=a,
                                       acqfn=acqfn)
    next_exp_id = []
    del gp_model, X_norm, Y_norm, X, Y, X_pending

    if X_pending_norm is not None:
        for t in range(n_trust_regions):
            next_exp_id_m = []
            for j in range(batch_size):
                ids = np.where(
                    np.all(X_pending_norm.detach().numpy() == X_next[
                        t, j, :].detach().numpy().reshape(
                            1, X_pending_norm.shape[1]),
                           axis=1) == True)[0]
                next_exp_id_m.extend(ids)
            next_exp_id.append(next_exp_id_m)
        if verbose == True:
            print("Next experiment to pick: ",
                  X_next.detach().numpy(), "Acqusition value: ",
                  acq_value.detach().numpy())
        next_exp_id = np.vstack(next_exp_id)
    del X_pending_norm
    return X_next, acq_value, next_exp_id


def cluster_bo_design(X,
                      Y,
                      ncluster_grid,
                      X_pending=None,
                      gp_method='gp_regression',
                      cluster_method='gmm',
                      batch_size=1,
                      likelihood=None,
                      covar_module=None,
                      min_inferred_noise_level=1e-4,
                      random_state=0,
                      acqfn='ei',
                      normalize=True,
                      verbose=False):
    """Run experimental design using BO
    Parameters
    ----------
    X : pyTorch tensor with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation. 
    Y : pyTorch tensor with a shape of (n_training_samples, 1) of floats
        Current measurements using X experimental design. 
    ncluster_grid : list of ints
        List of possible number of clusters. 
    X_pending : pyTorch tensor with a shape of (n_pending_samples, feature_size) of floats
        Current search space of experimental design after featurziation. 
    gp_method : str, default='gp_regression'
        Regression method used in this run. Must be 'gp_regression' or 'robust_regression'
    cluster_method : str, default='gmm'
        Clustering method options. Must be 'gmm' or 'kmeans' now.
    batch_size : int, default=1
        Number of next experiments to be added
    likelihood : GpyTorch likelihood, default=None
        Likelihood used in GP
    covar_module: GpyTorch kernel module, default=None
        Kernel used in GP
    min_inferred_noise_level : float, default=1e-4
        Minimum value of added noises to kernel 
    random_state : int, default=0
        Random seed for GMM clustering.
    verbose : boolean, default=False
        Print out the details of selcted experiments
    **kwargs : options
        Additional options in sklearn.
    Returns
    -------
    X_next : pyTorch tensor with a shape of (batch_size, feature_size) of floats
        Selected experiments by BO.
    acq_value : pyTorch tensor with a shape of of shape (batch_size,) of floats
        Acqusition values for the selected experiments
    next_exp_id : list of indices of length of batch_size
        Indices of selected X with length of batch_size in pending X 
    References
    ----------
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, 
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian 
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    """
    from .bo import generate_batch
    if normalize:
        X_norm, Y_norm, X_pending_norm, stats = normalize_data(
            X, Y, X_pending=X_pending)
    else:
        X_norm, Y_norm, X_pending_norm = X, Y, X_pending

    if gp_method == 'robust_regression':
        model, inliers, outliers = RobustRegression(
            X_norm,
            Y_norm,
            covar_module=covar_module,
            min_inferred_noise_level=min_inferred_noise_level)
        X_norm, Y_norm = X_norm[inliers, :], Y_norm[inliers]
        if len(inliers) != len(Y):
            print(len(Y) - len(inliers), ' outliers found')

    if cluster_method == 'gmm':
        from .clustering import gmm
        cluster_model, scores = gmm(ncluster_grid, X, random_state)
    elif cluster_method == 'kmeans':
        from .clustering import kmeans
        cluster_model, scores = kmeans(ncluster_grid, X, random_state)
    else:
        warnings.warn(
            "'cluster_method' can only be 'gmm' or 'kmeans', switching to 'gmm' method now"
        )
        from .clustering import gmm
        cluster_method = 'gmm'
        cluster_model, scores = gmm(ncluster_grid, X, random_state)

    pred_train_labels = cluster_model.predict(X.numpy())
    pred_pending_labels = cluster_model.predict(X_pending.numpy())
    ncluster = ncluster_grid[np.argmin(scores)]
    X_next, acq_value_all, sele_id_all = [], [], []

    for i in range(ncluster):
        train_ids = np.where(pred_train_labels == i)[0]
        pending_ids = np.where(pred_pending_labels == i)[0]
        TRAIN_X, TRAIN_Y = X[train_ids, :], Y[train_ids]
        TEST_X = X_pending[pending_ids, :]
        feature_keeps = []
        for m in range(TRAIN_X.shape[1]):
            a = (TRAIN_X[:, m] - TRAIN_X[0, m]).numpy()
            if a.any() != 0:
                feature_keeps.append(m)
        TRAIN_X, TEST_X = TRAIN_X[:, feature_keeps], TEST_X[:, feature_keeps]
        if TEST_X.shape[0] > 1:
            X_next_dict, acqf_value_dict, next_exp_id_dict = bo_design(
                X=TRAIN_X,
                Y=TRAIN_Y,
                X_pending=TEST_X,
                gp_method=gp_method,
                batch_size=batch_size,
                likelihood=likelihood,
                covar_module=covar_module,
                min_inferred_noise_level=min_inferred_noise_level,
                acqfn=acqfn,
                verbose=verbose)
            acq_value_all.extend(acqf_value_dict.ravel())
            sele_id_all.extend(pending_ids[next_exp_id_dict])
        else:
            acq_value_all.append(np.NINF)
            sele_id_all.append(-1)
    if list(set(sele_id_all)) == [-1]:
        warnings.warn("No possible candidate is selected in this iteration")
        return np.nan, np.nan, np.nan
    else:
        sort_ids = np.argsort(acq_value_all)
        next_exp_id = np.array(sele_id_all, dtype=int)[sort_ids[-batch_size:]]
        acq_value = np.array(acq_value_all)[sort_ids[-batch_size:]]
        X_next = X_pending[next_exp_id, :]
        return X_next, acq_value[sort_ids[-batch_size:]], next_exp_id


def cluster_turbo_design(state,
                         X,
                         Y,
                         ncluster_grid,
                         n_trust_regions=1,
                         X_pending=None,
                         gp_method='gp_regression',
                         cluster_method='gmm',
                         batch_size=1,
                         likelihood=None,
                         covar_module=None,
                         min_inferred_noise_level=1e-4,
                         random_state=0,
                         acqfn='ei',
                         normalize=True,
                         verbose=False):
    """Run experimental design using BO
    Parameters
    ----------
    X : pyTorch tensor with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation. 
    Y : pyTorch tensor with a shape of (n_training_samples, 1) of floats
        Current measurements using X experimental design. 
    ncluster_grid : list of ints
        List of possible number of clusters. 
    X_pending : pyTorch tensor with a shape of (n_pending_samples, feature_size) of floats
        Current search space of experimental design after featurziation. 
    gp_method : str, default='gp_regression'
        Regression method used in this run. Must be 'gp_regression' or 'robust_regression'
    cluster_method : str, default='gmm'
        Clustering method options. Must be 'gmm' or 'kmeans' now.
    batch_size : int, default=1
        Number of next experiments to be added
    likelihood : GpyTorch likelihood, default=None
        Likelihood used in GP
    covar_module: GpyTorch kernel module, default=None
        Kernel used in GP
    min_inferred_noise_level : float, default=1e-4
        Minimum value of added noises to kernel 
    random_state : int, default=0
        Random seed for GMM clustering.
    verbose : boolean, default=False
        Print out the details of selcted experiments
    **kwargs : options
        Additional options in sklearn.
    Returns
    -------
    X_next : pyTorch tensor with a shape of (batch_size, feature_size) of floats
        Selected experiments by BO.
    acq_value : pyTorch tensor with a shape of of shape (batch_size,) of floats
        Acqusition values for the selected experiments
    next_exp_id : list of indices of length of batch_size
        Indices of selected X with length of batch_size in pending X 
    References
    ----------
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, 
    and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian 
    Optimization. Advances in Neural Information Processing Systems 33, 2020.
    """
    from .turbo import generate_batch
    if normalize:
        X_norm, Y_norm, X_pending_norm, stats = normalize_data(
            X, Y, X_pending=X_pending)
    else:
        X_norm, Y_norm, X_pending_norm = X, Y, X_pending

    if gp_method == 'robust_regression':
        model, inliers, outliers = RobustRegression(
            X_norm,
            Y_norm,
            covar_module=covar_module,
            min_inferred_noise_level=min_inferred_noise_level)
        X_norm, Y_norm = X_norm[inliers, :], Y_norm[inliers]
        if len(inliers) != len(Y):
            print(len(Y) - len(inliers), ' outliers found')

    if cluster_method == 'gmm':
        from .clustering import gmm
        cluster_model, scores = gmm(ncluster_grid, X, random_state)
    elif cluster_method == 'kmeans':
        from .clustering import kmeans
        cluster_model, scores = kmeans(ncluster_grid, X, random_state)
    else:
        warnings.warn(
            "'cluster_method' can only be 'gmm' or 'kmeans', switching to 'gmm' method now"
        )
        from .clustering import gmm
        cluster_method = 'gmm'
        cluster_model, scores = gmm(ncluster_grid, X, random_state)

    pred_train_labels = cluster_model.predict(X.numpy())
    pred_pending_labels = cluster_model.predict(X_pending.numpy())
    ncluster = ncluster_grid[np.argmin(scores)]
    acq_value_all = np.NINF * np.ones((ncluster * batch_size, n_trust_regions))
    sele_id_all = -1 * np.ones(
        (ncluster * batch_size, n_trust_regions), dtype=int)

    for i in range(ncluster):
        train_ids = np.where(pred_train_labels == i)[0]
        pending_ids = np.where(pred_pending_labels == i)[0]
        TRAIN_X, TRAIN_Y = X[train_ids, :], Y[train_ids]
        TEST_X = X_pending[pending_ids, :]
        feature_keeps = []
        for m in range(TRAIN_X.shape[1]):
            a = (TRAIN_X[:, m] - TRAIN_X[0, m]).numpy()
            if a.any() != 0:
                feature_keeps.append(m)
        TRAIN_X, TEST_X = TRAIN_X[:, feature_keeps], TEST_X[:, feature_keeps]
        if state.restart_triggered == False and TEST_X.shape[0] > 1:
            X_next_dict, acq_value_dict, next_exp_id_dict = turbo_design(
                state=state,
                X=TRAIN_X,
                Y=TRAIN_Y,
                n_trust_regions=n_trust_regions,
                X_pending=TEST_X,
                gp_method=gp_method,
                batch_size=batch_size,
                likelihood=likelihood,
                covar_module=covar_module,
                min_inferred_noise_level=min_inferred_noise_level,
                acqfn=acqfn,
                verbose=verbose)
            acq_value_all[i * batch_size:(i + 1) *
                          batch_size, :] = acq_value_dict.detach().numpy().T
            sele_id_all[i * batch_size:(i + 1) *
                        batch_size, :] = np.array(next_exp_id_dict).T
    if list(set(sele_id_all.ravel())) == [-1]:
        warnings.warn("No possible candidate is selected in this iteration")
        return np.nan, np.nan, np.nan
    else:
        sort_ids = np.argsort(acq_value_all, axis=0)
        next_exp_id = np.array(sele_id_all)[
            sort_ids[-batch_size:],
            np.repeat(np.arange(n_trust_regions)[
                None, :], batch_size, axis=0)].T
        acq_value = np.array(acq_value_all)[
            sort_ids[-batch_size:],
            np.repeat(np.arange(n_trust_regions)[
                None, :], batch_size, axis=0)].T
        X_next = X_pending[next_exp_id, :]
        return X_next, torch.tensor(acq_value), next_exp_id
