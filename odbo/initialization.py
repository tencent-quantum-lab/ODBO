"""Initialization algorithm to find the 0th round experiments for BO"""

import numpy as np


def initial_design(X_pending,
                   choice_list=None,
                   least_occurance=None,
                   verbose=True,
                   allow_abundance=False,
                   importance_method='sum',
                   update_method='independent'):
    """Generate the inital set of experimentes to collect measurements to initiate BO  
    Parameters
    ----------
    X_pending : ndarray with shape of (n_pending_samples, feature_size)
        Current search space of experimental design after featurziation. 
        Features can be ints, floats and strings
    choice_list : list of list, default=None
        List of list of choices for each feature. Length of the list is the number of features
    least_occurance : List of ints or an array of ints, default=None
        Least occurance of each choice for each feature.
    verbose : boolen, deafult=True,
        Printout the current selected experiments
    allow_abundance : boolen, deafult=True,
        If allow using the abundance scores to pick less frequent choices first
    importance_method : string, default='sum'
        The importance score computation method

    Returns
    -------
    sele_indices : List of ints
        Selected indices of the input X_pending to be 0th round experiments for BO
    """

    if least_occurance is None:
        least_occurance = np.ones(X_pending.shape[1])
    if choice_list is None:
        choice_list = []
        for i in range(X_pending.shape[1]):
            choice_list.append(list(set(X_pending[:, i])))
    N = X_pending.shape[0]
    if allow_abundance:
        abundance_scores = abundance(X_pending, choice_list)
    pending_scores = np.zeros(X_pending.shape)
    for i in range(X_pending.shape[1]):
        ids = np.where(X_pending[:, i] == '*')[0]
        if len(ids) != 0:
            X_pending[ids, i] = X_pending[0, i]
    sele_indices = [0]
    pending_indices = np.arange(1, N)
    pending_scores[sele_indices, :] = -np.inf * np.ones(
        pending_scores[sele_indices, :].shape)
    pending_scores[pending_indices, :] = compute_score(
        X_pending[sele_indices, :], X_pending[pending_indices, :], choice_list,
        least_occurance, importance_method)
    if verbose == True:
        print('Current selected experiments: ', sele_indices[-1],
              'Max pending score: ', np.max(pending_scores))

    while True:
        if allow_abundance:
            sum_scores = np.sum(
                np.multiply(pending_scores, abundance_scores), axis=1)
        else:
            sum_scores = np.sum(pending_scores, axis=1)
        sum_scores[sele_indices] = -np.inf * np.ones(len(sele_indices))
        if np.max(pending_scores) <= 0.0:
            break
        update_indices = np.argmax(sum_scores)
        sele_indices.append(update_indices)
        pending_scores = update_score(pending_scores,
                                      X_pending[update_indices, :], X_pending,
                                      update_method)
        if verbose == True:
            print('Current selected experiments: ', sele_indices[-1],
                  'Max pending score: ', np.max(pending_scores))
    return sele_indices


def compute_score(current_X,
                  X_pending,
                  choice_list,
                  least_occurance,
                  sele_indices,
                  importance_method='sum'):
    """Comput scores of all the experiments in the search space
    Parameters
    ----------
    current_X : ndarray with shape of (n_current_samples, feature_size)
        Current selected experiments
    X_pending : ndarray with shape of (n_pending_samples, feature_size)
        Current search space of experimental design after featurziation. 
        Features can be ints, floats and strings
    choice_list : list of list, default=None
        List of list of choices for each feature. Length of the list is the number of features.
    least_occurance : List of ints or an array of ints, default=None
        Least occurance of each choice for each feature.
    importance_method : string, default='sum'
        The importance score computation method

    Returns
    -------
    scores : ndarray of (n_pending_samples, feature_size)  of floats
        Importances scores of each choice of each feature
    """

    scores = np.zeros(X_pending.shape)
    if importance_method == 'sum':
        for i in range(X_pending.shape[1]):
            for j in range(len(choice_list[i])):
                current_id_no = np.where(
                    current_X[:, i] == choice_list[i][j])[0]
                pending_id_no = np.where(
                    X_pending[:, i] == choice_list[i][j])[0]
                raw_score = least_occurance[i] - len(current_id_no)
                scores[pending_id_no, i] = raw_score
        return scores
    else:
        print(
            'Other importance score computation method is not implemented yet.'
        )


def update_score(pending_scores,
                 update_X,
                 X_pending,
                 update_method='independent'):
    """Update scores of all the experiments with knowing the newly selected experiments
    Parameters
    ----------
    pending_scores : ndarray with shape of (n_pending_samples, feature_size) of floats
        Scores from last round of selection
    update_X : ndarray with shape of (1, feature_size)
        Newly selected experiment
    X_pending : ndarray with shape of (n_pending_samples, feature_size)
        Current search space of experimental design after featurziation. 
        Features can be ints, floats and strings
    Returns
    -------
    pending_scores : ndarray of (n_pending_samples, feature_size)
        Updated importances scores of each choice of each feature
    """
    if update_method == 'independent':
        for i in range(X_pending.shape[1]):
            pending_id_no = np.where(X_pending[:, i] == update_X[i])[0]
            pending_scores[pending_id_no,
                           i] = pending_scores[pending_id_no, i] - 1
    elif update_method == 'correlate':
        wild_type = X_pending[0, :]
        diff_loc = np.where(wild_type != update_X)[0]
        for i in range(X_pending.shape[1]):
            for j in range(len(diff_loc)):
                pending_id_no = np.where(
                    np.logical_and(X_pending[:, i] == update_X[diff_loc[j]],
                                   wild_type[i] == wild_type[diff_loc[j]]))[0]
                pending_scores[pending_id_no,
                               i] = pending_scores[pending_id_no, i] - 1
    return pending_scores


def abundance(X_pending, choice_list):
    """Abudnace scores of the experiments 
    Parameters
    ----------
    X_pending : ndarray with shape of (n_pending_samples, feature_size)
        Current search space of experimental design after featurziation. 
        Features can be ints, floats and strings. 
    choice_list : list of list, default=None
        List of list of choices for each feature. Length of the list is number of features.
    Returns
    -------
    abundance : ndarray of (n_pending_samples, feature_size)
        Abundance scores using the frequency of each choice for each feature.
    """

    N, feature_size = X_pending.shape[0], X_pending.shape[1]
    abundance = np.zeros(X_pending.shape)
    for i in range(feature_size):
        for j in range(len(choice_list[i])):
            pending_id_no = np.where(X_pending[:, i] == choice_list[i][j])[0]
            abundance[pending_id_no, i] = (
                N / len(choice_list[i]) / len(pending_id_no))**2 * np.ones(
                    len(pending_id_no))
    return abundance
