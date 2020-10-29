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
from sklearn.metrics import accuracy_score


# new class to mock the forecast_model class
class MockForecastModel(BaseEstimator, ClassifierMixin):

    def __init__(self, model_path: str):
        """
        :param model_path: path to model decision trees.
        """
        self.model_path = model_path
        self.trees = []
        self.feats = []
        self.feats_idx = []

        # classes_ gives the classes in numerical form
        # means that the first column in predict_prob is associated with 0 and second is with 1
        self.classes_ = np.array([0, 1])

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
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):

        if not self.feats_idx:
            # check to make sure feats_idx array has been initialised
            raise AssertionError("self.feats_idx is empty. Run prepare_for_calibration method.")

        if isinstance(X, pd.DataFrame):
            # convert to 2D numpy array
            X = X.to_numpy()

        # initialise probability matrix
        y_proba = np.zeros((X.shape[0], 2))

        for idx, tree in enumerate(self.trees):
            # use features from feature files when predicting so they match the features used for training
            pred = tree.predict(X[:, self.feats_idx[idx]])

            # turn the predictions into indices 0 and 1 by converting them to integer
            pred = list(map(int, np.round(pred)))
            # add a count to the classification made for each observation
            for i, p in enumerate(pred):
                y_proba[i, p] += 1

        # turning counts into probabilities
        y_proba = y_proba/len(self.trees)

        return y_proba

    def prepare_for_calibration(self, X):
        # save the indices of columns corresponding to the features used in the trees
        self.feats_idx = []
        for features in self.feats:
            self.feats_idx.append([X.columns.get_loc(f) for f in features])

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


def calibration(download_data=False, plots=True):
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

    # save the indices of the columns corresponding to features for each tree
    classifier.prepare_for_calibration(X)

    calibrated_classifier = CalibratedClassifierCV(classifier,  method='sigmoid', cv='prefit')
    calibrated_classifier.fit(X, y['label'])

    if plots:
        # ==== plot of calibrated probabilities vs thresholds ====
        f, ax = plt.subplots(1, 1, figsize=(18, 12))

        plt.axvline(0.8, color='pink', linewidth=5, zorder=1)
        plt.scatter(classifier.predict_proba(X)[:, 1], calibrated_classifier.predict_proba(X)[:, 1],
                    c='orange', s=60, zorder=2)
        # plt.axhline(0.086, color='black', xmax=0.82, linewidth=2, zorder=1, ls='--')

        plt.xlabel("Threshold for predicting eruption", fontsize=25)
        plt.ylabel("Probability of eruption", fontsize=25)
        plt.title("Threshold vs probability of eruption", fontsize=40)
        for t in ax.get_xticklabels() + ax.get_yticklabels():  # increase of x and y tick labels
            t.set_fontsize(20.)

        os.makedirs(fm.plotdir, exist_ok=True)
        plt.savefig(f"{fm.plotdir}/threshold_vs_probability.png", format='png', dpi=300)
        plt.close()

        # ==== plot of actual target vector, predicted consensus values and calibrated probabilities ====
        forecast = fm.forecast(ti=te - month, tf=te + month, recalculate=True)
        f, ax = plt.subplots(3, 1, figsize=(18, 12), sharex="all")

        ax[0].plot(X.index, y['label'], linewidth=3)
        ax[0].axvline(te, color='pink', linewidth=3)
        ax[0].set_ylabel("Actual binary target", fontsize=17)
        ax[1].plot(X.index, forecast['consensus'], linewidth=3)
        ax[1].axvline(te, color='pink', linewidth=3)
        ax[1].set_ylabel("Prediction Consensus", fontsize=17)
        ax[2].plot(X.index, calibrated_classifier.predict_proba(X)[:, 1], linewidth=3)
        ax[2].axvline(te, color='pink', linewidth=3)
        ax[2].set_ylabel("Calibrated Probability", fontsize=17)
        ax[2].set_xlabel("Time", fontsize=17)
        f.suptitle("Actual Target, Prediction Consensus and Calibrated Probabilities", fontsize=40.)
        # increase x and y tick labels
        for t in ax[0].get_yticklabels() + ax[1].get_yticklabels() + ax[2].get_yticklabels() + ax[2].get_xticklabels():
            t.set_fontsize(15.)

        os.makedirs(fm.plotdir, exist_ok=True)
        plt.savefig(f"{fm.plotdir}/actual_predictions_probabilities.png", format='png', dpi=300)
        plt.close()


if __name__ == '__main__':
    os.chdir('..')  # set working directory to root
    calibration()
