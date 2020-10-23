import joblib
import os
from glob import glob

import matplotlib.pyplot as plt
from whakaari import TremorData, ForecastModel
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from time import time


# new class to mock the forecast_model class
class MockForecastModel(BaseEstimator, ClassifierMixin):

    def __init__(self, model_path: str):
        """

        :param model_path: path to model decision trees.
        """
        self.model_path = model_path
        self.trees = []
        self.feats = []
        self._read_trees()

    def _read_trees(self):
        # read in the trees and save to self.trees
        tree_files = glob(f"{self.model_path}\\*.pkl")
        for tree in tree_files:
            self.trees.append(joblib.load(tree))

        # read in the feature files for each tree
        tree_feat_files = glob(f"{self.model_path}\\[0-9]*.fts")
        for feat_file in tree_feat_files:
            f = open(feat_file, 'r')
            feats = list(map(lambda x: x.strip().split(' ', 1)[1], f.readlines()))
            f.close()
            self.feats.append(feats)

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
        y_proba = np.zeros((X.shape[0], 2))

        for idx, tree in enumerate(self.trees):
            # use features from feature files when predicting so they match the features used for training
            pred = tree.predict(X[self.feats[idx]])

            # turn the predictions into indices 0 and 1 by converting them to integer
            pred = list(map(int, np.round(pred)))
            # add a count to the classification made for each observation
            for i, p in enumerate(pred):
                y_proba[i, p] += 1

        # turning counts into probabilities
        y_proba = y_proba/len(self.trees)

        return y_proba


def calibration(download_data=False):
    # constants
    month = timedelta(days=365.25 / 12)

    td = TremorData()
    # QUESTION is this needed???
    if download_data:
        for te in td.tes:
            td.update(ti=te-month, tf=te+month)

    # construct model object
    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root='calibration_forecast_model')

    # columns to manually drop from feature matrix because they are highly correlated to other
    # linear regressors
    drop_features = ['linear_trend_timewise', 'agg_linear_trend']

    # set the available CPUs higher or lower as appropriate
    n_jobs = 6

    # train the model, excluding 2019 eruption
    # note: building the feature matrix may take several hours, but only has to be done once
    # and will intermittantly save progress in ../features/
    # trained scikit-learn models will be saved to ../models/*root*/
    te = td.tes[-1]
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, retrain=True,
             exclude_dates=[[te - month, te + month], ], n_jobs=n_jobs)

    classifier = MockForecastModel(fm.modeldir)

    # get feature and labels for eruption not used in training
    X, y = fm._extract_features(ti=te - month, tf=te + month)

    cccv = CalibratedClassifierCV(classifier, cv='prefit')
    cccv.fit(X, y['label'])
    # TODO your_cccv.fit(X_validation, y_validation) <- This validation data is used solely for calibration purposes.


if __name__ == '__main__':
    os.chdir('..')  # set working directory to root
    calibration()
