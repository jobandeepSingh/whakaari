import joblib
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV


# new class to mock the forecast_model class
class MockForecastModel(BaseEstimator, ClassifierMixin):

    def __init__(self, model_path: str = "models"):
        """

        :param model_path: path to model decision trees, relative to root of repo.
            Defaults to models directory.
        """
        self.model_path = model_path
        self.trees = []
        self._read_trees()

    def _read_trees(self):
        tree_files = glob(f"{path}\\*.pkl")
        for tree in tree_files:
            self.trees.append(joblib.load(tree))

    def fit(self, X, y):
        """
        Mock fit function. Does nothing.

        :param X: numpy array of shape [n_samples, n_features]
            Training set.
        :param y:  numpy array of shape [n_samples]
            Target values.
        :return:
        """

        # this class is only for calibration
        # so this is empty
        pass

    def predict(self, X):
        """
        :param X: array-like of shape (n_samples, n_features)
            Test samples.
        :return: y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted labels for X.
        """
        pass

    def predict_proba(self, X):
        # initialise probability matrix, shape = nx2
        y_proba = np.zeros((len(self.trees), 2))

        for tree in self.trees:
            pred = tree.predict(X)
            # turn the predictions into indices 0 and 1 by converting them to integer
            pred = list(map(int, np.round(pred)))
            # add a count to the classification made for each observation
            for i, p in enumerate(pred):
                y_proba[i, p] += 1

        # turning counts into probabilities
        y_proba = y_proba/len(self.trees)

        return y_proba


if __name__ == '__main__':
    os.chdir('..')  # set working directory to root

    # path to model
    path = "models\\fm_2.00wndw_0.75ovlp_2.00lkfd_dsar-hf-mf-rsam"

    classifier = MockForecastModel(path)

    # TODO need to read in actual data here
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 20)), columns=list('ABCDEFGHIJKLMNOPRSTU'))
    classifier.predict_proba(df)

    cccv = CalibratedClassifierCV(classifier, cv='prefit')
    cccv.fit()  # TODO add out of sample X and y as input
    # TODO your_cccv.fit(X_validation, y_validation) <- This validation data is used solely for calibration purposes.

    pass
