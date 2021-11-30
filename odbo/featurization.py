import numpy as np
import warnings


class FeatureTransform(object):
    """Abstract base class for any FeatureTransformation method
    """

    def __init__(self, raw_vars, cat_features=None, Y=None, categories=None):
        """Constructor for the FeatureTransform base class.
        Args:
            raw_vars : Input experiments expressed using raw variable names
            cat_features : Featurizations of different categories of each variable
            Y : Input training measurements
            categories : Categories/Choices of each variable
        """
        self._raw_vars = raw_vars
        self._cat_features = cat_features
        self._Y = Y
        self._catergories = categories

    def transform(self, raw_vars):
        """Transform input experiments to standard encodings
        Args:
            raw_vars : Input experiments expressed using raw variable names        
        """
        transformed_feature = np.ones(raw_vars.shape) * np.nan
        for i in range(raw_vars.shape[1]):
            for j in range(len(self._categories[i])):
                ids = [
                    k for k, x in enumerate(raw_vars[:, i])
                    if x == self._categories[i][j]
                ]
                transformed_feature[ids, i] = self._cat_features[i][j]
                try:
                    np.isnan(transformed_feature.sum())
                except InputError as err:
                    print('InputError: A wrong experimental variable at ',
                          np.argwhere(np.isnan(transformed_feature)))
        return transformed_feature


class AvgMeasurement(FeatureTransform):
    """ AvgMeasurement class using the average of the measurements 
    of experiments having the apperence of a certain choice for a variable as 
    the feature
    """

    def __init__(self, raw_vars, Y, cat_features=None, categories=None):
        super(AvgMeasurement, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories)
        self._name = 'AvgMeasurement'
        if categories == None:
            categories = []
            for i in range(raw_vars.shape[1]):
                choice = list(set(raw_vars[:, i]))
                categories.append(choice)
        self._categories = categories
        if cat_features == None:
            cat_features = []
            for i in range(raw_vars.shape[1]):
                feature_choice = np.empty(len(categories[i]))
                for j in range(len(categories[i])):
                    ids = [
                        k for k, x in enumerate(raw_vars[:, i])
                        if x == categories[i][j]
                    ]
                    feature_choice[j] = np.mean(Y[ids])
                cat_features.append(feature_choice)
        self._cat_features = cat_features


class RankMeasurement(FeatureTransform):
    """ RankMeasurement class using the rank of the average of the measurements 
    of experiments having the apperence of a certain choice for a variable as 
    the feature
    """

    def __init__(self, raw_vars, Y, cat_features=None, categories=None):
        super(RankMeasurement, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories)
        self._name = 'RankMeasurement'
        if categories == None:
            categories = []
            for i in range(raw_vars.shape[1]):
                choice = list(set(raw_vars[:, i]))
                categories.append(choice)
        self._categories = categories
        if cat_features == None:
            cat_features = []
            for i in range(raw_vars.shape[1]):
                feature_choice = np.empty(len(categories[i]))
                for j in range(len(categories[i])):
                    ids = [
                        k for k, x in enumerate(raw_vars[:, i])
                        if x == categories[i][j]
                    ]
                    feature_choice[j] = np.mean(Y[ids])
                    ranks = np.argsort(feature_choice)
                cat_features.append(ranks)
        self._cat_features = cat_features


class MaxMeasurement(FeatureTransform):
    """ MaxMeasurement class using max value of the measurements 
    of experiments having the apperence of a certain choice for a variable as 
    the feature
    """

    def __init__(self, raw_vars, Y, cat_features=None, categories=None):
        super(MaxMeasurement, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories)
        self._name = 'MaxMeasurement'
        if categories == None:
            categories = []
            for i in range(raw_vars.shape[1]):
                choice = list(set(raw_vars[:, i]))
                categories.append(choice)
        self._categories = categories
        if cat_features == None:
            cat_features = []
            for i in range(raw_vars.shape[1]):
                feature_choice = np.empty(len(categories[i]))
                for j in range(len(categories[i])):
                    ids = [
                        k for k, x in enumerate(raw_vars[:, i])
                        if x == categories[i][j]
                    ]
                    feature_choice[j] = np.max(Y[ids])
                cat_features.append(feature_choice)
        self._cat_features = cat_features


class MinMeasurement(FeatureTransform):
    """ MinMeasurement class using min value of the measurements 
    of experiments having the apperence of a certain choice for a variable as 
    the feature
    """

    def __init__(self, raw_vars, Y, cat_features=None, categories=None):
        super(MinMeasurement, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories)
        self._name = 'MinMeasurement'
        if categories == None:
            categories = []
            for i in range(raw_vars.shape[1]):
                choice = list(set(raw_vars[:, i]))
                categories.append(choice)
        self._categories = categories
        if cat_features == None:
            cat_features = []
            for i in range(raw_vars.shape[1]):
                feature_choice = np.empty(len(categories[i]))
                for j in range(len(categories[i])):
                    ids = [
                        k for k, x in enumerate(raw_vars[:, i])
                        if x == categories[i][j]
                    ]
                    feature_choice[j] = np.min(Y[ids])
                cat_features.append(feature_choice)
        self._cat_features = cat_features
