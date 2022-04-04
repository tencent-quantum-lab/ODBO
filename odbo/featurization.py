import numpy as np
import warnings


class MassiveFeatureTransform(object):
    """MassiveFeatureTransform method for saturation mutagenesis at k positions
    """

    def __init__(self,
                 raw_vars,
                 cat_features=None,
                 Y=None,
                 categories=None,
                 method='Avg',
                 mode='independent'):
        """Constructor for the MassiveFeatureTransform class.
        Args:
            raw_vars : Input experiments expressed using raw variable names
            cat_features : Featurizations of different categories of each variable
            Y : Input training measurements
            categories : Categories/Choices of each variable
            method : Method to use the measurements as features, must be 'Avg', 'Max',
                     'Min', 'Rank'
            mode : Mode to infer features at different varible locations. 'independent' 
                   means varibles independently vary, 'correlate' means all the variables 
                   share the same features if the experimental choices are the same
        """

        self._raw_vars = raw_vars
        self._cat_features = cat_features
        self._Y = Y
        self._catergories = categories
        self._mode = mode
        self._method = method

        if categories == None:
            categories = []
            for i in range(raw_vars.shape[1]):
                if mode == 'independent':
                    choice = list(set(raw_vars[:, i]))
                elif mode == 'correlate' or mode == 'hybrid':
                    choice = list(set(raw_vars.ravel()))
                categories.append(choice)
        for i in range(self._raw_vars.shape[1]):
            ids = np.where(self._raw_vars[:, i]== '*')[0]
            self._raw_vars[ids, i] = self._raw_vars[0, i]

        self._categories = categories
        if cat_features == None:
            cat_features = []
            for i in range(raw_vars.shape[1]):
                feature_choice = np.empty(len(categories[i]))
                for j in range(len(categories[i])):
                    if mode == 'independent' or mode == 'hybrid':
                        ids = np.where(raw_vars[:, i] == categories[i][j])[0]
                        if len(ids) == 0:
                            ids = []
                            for t in range(raw_vars.shape[1]):
                                ids.extend(
                                    list(
                                        np.where(
                                            np.logical_and(
                                                raw_vars[:, t] == categories[i][j],
                                                raw_vars[0, t] == raw_vars[0, i]))
                                            [0]))
                    elif mode == 'correlate':
                        ids = []
                        for t in range(raw_vars.shape[1]):
                            ids.extend(
                                list(
                                    np.where(
                                        np.logical_and(
                                            raw_vars[:, t] == categories[i][j],
                                            raw_vars[0, t] == raw_vars[0, i]))
                                    [0]))
                    if len(ids) != 0:
                        if self._method == 'Avg':
                            feature_choice[j] = np.mean(Y[ids])
                        elif self._method == 'Max':
                            feature_choice[j] = np.max(Y[ids])
                        elif self._method == 'Min':
                            feature_choice[j] = np.min(Y[ids])
                    else:
                        feature_choice[j] = j
                cat_features.append(feature_choice)

        self._cat_features = cat_features

    def transform(self, raw_vars):
        """Transform input experiments to standard encodings
        Args:
            raw_vars : Input experiments expressed using raw variable names        
        """
        for i in range(raw_vars.shape[1]):
            ids = np.where(raw_vars[:, i]== '*')[0]
            raw_vars[ids, i] = self._raw_vars[0, i]
        transformed_feature = np.ones(raw_vars.shape) * np.nan
        for i in range(raw_vars.shape[1]):
            for j in range(len(self._categories[i])):
                ids = np.where(raw_vars[:, i] == self._categories[i][j])[0]
                transformed_feature[ids, i] = self._cat_features[i][j]
                try:
                    np.isnan(transformed_feature.sum())
                except InputError as err:
                    print('InputError: A wrong experimental variable at ',
                          np.argwhere(np.isnan(transformed_feature)))
        return transformed_feature


class FewFeatureTransform(MassiveFeatureTransform):
    """FewChangeMeasurement method for non-saturation mutagenesis
    """

    def __init__(self,
                 raw_vars,
                 cat_features=None,
                 Y=None,
                 categories=None,
                 max_change_length=None,
                 method='Avg',
                 mode='independent'):
        """Constructor for the FewFeatureTransform class.
        Args:
            raw_vars : Input experiments expressed using raw variable names
            cat_features : Featurizations of different categories of each variable
            Y : Input training measurements
            categories : Categories/Choices of each variable
            max_change_length : Maximum numbers of varibles changing in an experiment
            method : Method to use the measurements as features, must be 'Avg', 'Max',
                     'Min', 'Rank'
            mode : Mode to infer features at different varible locations. 'independent' 
                   means varibles independently vary, 'correlate' means all the variables 
                   share the same features if the experimental choices are the same
        """
        super(FewFeatureTransform, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories,
            mode=mode,
            method=method)
        self._max_change_length = max_change_length
        if self._max_change_length == None:
            self._max_change_length = 0
            for i in range(1, raw_vars.shape[0]):
                curr_len = len(np.where(raw_vars[0, :] != raw_vars[i, :])[0])
                if curr_len >= self._max_change_length:
                    self._max_change_length = curr_len


    def transform(self, raw_vars):
        """Transform input experiments to standard encodings
        Args:
            raw_vars : Input experiments expressed using raw variable names        
        """
        for i in range(raw_vars.shape[1]):
            ids = np.where(raw_vars[:, i]== '*')[0]
            raw_vars[ids, i] = self._raw_vars[0, i]
        for i in range(raw_vars.shape[0]):
            curr_len = len(np.where(self._raw_vars[0, :] != raw_vars[i, :])[0])
            if curr_len > self._max_change_length:
                warnings.warn(
                    "Entire search space changes more varibles per experiment. Need to retransform the training space"
                )
                self._max_change_length = curr_len

        transformed_feature = -np.ones((raw_vars.shape[0],self._max_change_length*2))

        for i in range(raw_vars.shape[0]):
            loc_change = np.where(raw_vars[i,:] != self._raw_vars[0,:])[0]
            if len(loc_change) ==0:
                transformed_feature[i, self._max_change_length:] =  np.ones(self._max_change_length)*self._Y[0]
            else:
                transformed_feature[i, 0:len(loc_change)] = loc_change
                for j in range(len(loc_change)):
                    feat = np.where(np.array(self._categories[loc_change[j]]) == raw_vars[i,loc_change[j]])[0]
                    transformed_feature[i, self._max_change_length+j] =  self._cat_features[loc_change[j]][feat]
                if len(loc_change) != self._max_change_length:
                    transformed_feature[i, self._max_change_length+len(loc_change):] =  np.ones(self._max_change_length-len(loc_change))*self._Y[0]

        return transformed_feature
