import numpy as np
import pandas as pd
import utils
import featurization, initialization


def test_avg_measurement():
    """ Test function for featurization
    """
    data_test = pd.read_csv('../datasets/GB1_2016_384.csv', sep=',')
    name_pre, Y_test = np.array(data_test['AACombo']), np.array(
        data_test['Fitness'])
    name = utils.code_to_array(name_pre)
    model = featurization.AvgMeasurement(raw_vars=name, Y=Y_test)
    trans_feat = model.transform(name[0:3, :])
    assert np.max(
        np.abs(trans_feat - np.array(
            [[0.20793691, 0.73780724, 0.18531278, 0.0720712],
             [1.05293156, 0.8985004, 1.31791446, 0.34846399],
             [0.00479801, 0.8985004, 0.2116735, 0.00232793]]))) <= 1e-6


def test_initial_design():
    """ Test function for initialization
    """
    np.random.seed(0)
    X_pending = np.empty((500, 5))
    for i in range(5):
        X_pending[:, i] = np.random.choice(range(7), 500)
    sele = initialization.initial_design(X_pending, verbose=False)


test_avg_measurement()
test_initial_design()
