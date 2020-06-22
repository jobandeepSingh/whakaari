from scripts.TremorData import TremorData, _DAY, makedir, DEFAULT_SECS_BETWEEN_OBS, read_hdh5, \
    write_to_hdf5, datetimeify, is_in_store
from datetime import timedelta, datetime
from tsfresh.transformers import FeatureSelector
import pandas as pd
import numpy as np
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from typing import List
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from inspect import getfile, currentframe
import os
from itertools import chain


class RegressionModel(object):
    def __init__(self, window: int, period_before: int, overlap: float = 0., period_after: int = None,
                 data_streams: List[str] = ['rsam', 'mf', 'hf', 'dsar'], root: str = None, n_jobs: int = 6,
                 sec_between_obs: int = DEFAULT_SECS_BETWEEN_OBS):
        """
        Initialises RegressionModel Object

        :param window: Length of window in seconds
        :param period_before: Period (in hours) before an eruption to consider
        :param overlap: Fraction of overlap between adjacent windows. Set this to 1. for overlap of entire
                        window minus 1 data point.
        :param period_after: Period (in hours) after an eruption to consider
        :param data_streams: Data streams and transforms from which to extract features.
        :param root:
        """
        self.n_jobs = n_jobs
        self.window = window
        self.overlap = overlap
        self.period_before = timedelta(hours=period_before)
        if period_after is None:
            period_after = period_before
        self.period_after = timedelta(hours=period_after)
        # ep - eruptive period
        self.ep_length = self.period_before + self.period_after + timedelta(days=1.)
        self.data_streams = data_streams
        # sbo - seconds between observations
        self.sbo = sec_between_obs

        # make Tremor Data object and update it to include data
        # around each eruption
        self.data = TremorData()
        # eps - eruptive periods
        self.eps = []
        for erp in self.data.tes:
            ep = []
            for i in range(self.ep_length.days):
                ep.append(erp - self.period_before + _DAY * i)
            self.eps.append(ep)

        for erp in self.data.tes:
            self.data.update(dt_floor(erp - self.period_before), dt_ceil(erp + self.period_after), self.sbo)

        # dtw - delta time window
        self.dtw = timedelta(seconds=self.window)
        self.dt = timedelta(seconds=self.sbo)  # timedelta(seconds=600)  # FIXME this variable need to be dynamic too!!
        # dto - delta time non-overlapping section of windows
        self.dto = (1. - self.overlap) * self.dtw
        # iw - number of observations in a window
        self.iw = int(self.window / self.sbo)
        # iw - number of observations in overlapping section of a window
        self.io = int(self.overlap * self.iw)
        if self.io == self.iw:
            self.io -= 1

        # QUESTION this should take care of decimal seconds window and sbo ?
        self.window = self.iw * self.sbo
        self.dtw = timedelta(seconds=self.window)
        self.overlap = self.io * 1. / self.iw
        self.dto = (1. - self.overlap) * self.dtw

        # QUESTION What is this for?
        self.update_feature_matrix = True

        self.fm = None
        self.ys = None

        # set the data to only the data needed for regression
        # FIXME possibly not needed
        # self.data.set_data([t-self.period_before for t in self.data.tes],
        #                    [t+self.period_after for t in self.data.tes])

        # add naming convention for files
        if root is None:
            self.root = 'fm_{:3.2f}wndw_{:3.2f}ovlp_{:3.2f}pb_{:3.2f}pa'.format(self.window, self.overlap,
                                                                                self.period_before.total_seconds(),
                                                                                self.period_after.total_seconds())
            self.root += '_' + (('{:s}-' * len(self.data_streams))[:-1]).format(*sorted(self.data_streams))
        else:
            self.root = root
        self.rootdir = os.sep.join(getfile(currentframe()).split(os.sep)[:-2])
        self.plotdir = f'{self.rootdir}/plots/{self.root}'
        self.modeldir = f'{self.rootdir}/models/{self.root}'
        self.featdir = f'{self.rootdir}/features'
        self.featfile = f'{self.featdir}/{self.root}_features.h5'
        self.preddir = f'{self.rootdir}/predictions/{self.root}'

    def _construct_windows(self, nw, ti, i0=0, i1=None):
        """ Create overlapping data windows for feature extraction.

            Parameters:
            -----------
            Nw : int
                Number of windows to create.
            ti : datetime.datetime
                End of first window.
            i0 : int
                Skip i0 initial windows.
            i1 : int
                Skip i1 final windows.

            Returns:
            --------
            df : pandas.DataFrame
                Dataframe of windowed data, with 'id' column denoting individual windows.
            window_dates : list
                Datetime objects corresponding to the beginning of each data window.
        """
        if i1 is None:
            i1 = nw

        # get data for windowing period
        df = self.data.get_data(ti - self.dtw, ti + (nw - 1) * self.dto)[self.data_streams]

        # create windows
        dfs = []
        for i in range(i0, i1):
            dfi = df[:].iloc[i * (self.iw - self.io):i * (self.iw - self.io) + self.iw]
            dfi['id'] = i + 1
            dfs.append(dfi)
        df = pd.concat(dfs)
        window_dates = [ti + (i + 1) * self.dto for i in range(nw)]  # i+1 to get the time at the end of the window
        return df, window_dates[i0:i1]

    def _extract_features(self, ti, tf, eruption=None, feats=None):
        """ Extract features from windowed data.

            Parameters:
            -----------
            ti : datetime.datetime
                End of first window.
            tf : datetime.datetime
                End of last window.

            Returns:
            --------
            fm : pandas.Dataframe
                tsfresh feature matrix extracted from data windows.
            ys : pandas.Dataframe
                Label vector corresponding to data windows

            Notes:
            ------
            # FIXME this needs to change
            Saves feature matrix to $rootdir/features/$root_features.csv to avoid recalculation.
        """
        makedir(self.featdir)

        # number of windows in feature request
        # Nw = int(np.floor(((tf - ti) / self.dt) / (self.iw - self.io)))
        Nw = int(np.floor(((tf - ti) / self.dt) / (self.iw - self.io)))

        # features to compute
        if feats == "comprehensive":
            cfp = ComprehensiveFCParameters()
        elif feats == "minimal":
            cfp = MinimalFCParameters()
        else:  # defaults to efficient
            feats = "efficient"
            cfp = EfficientFCParameters()
        # if self.compute_only_features:
        #     cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in self.compute_only_features])
        # else:
        #     # drop features if relevant
        #     _ = [cfp.pop(df) for df in self.drop_features if df in list(cfp.keys())]

        fp = f"/{ti}_{tf}_{feats}_features"
        # check if feature matrix already exists and what it contains
        if is_in_store(fp, self.featfile):
            # if os.path.isfile(self.featfile):
            # FIXME need to change to hdf5
            # t = pd.to_datetime(pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], usecols=['time'],
            #                                infer_datetime_format=True).index.values)
            feat_df = read_hdh5(self.featfile, fp)
            t = pd.to_datetime(feat_df.index.values)

            ti0, tf0 = t[0] - self.dtw, t[-1] - self.dtw  # -self.dtw as datetime of first window is end of window
            Nw0 = len(t)
            # FIXME need to change to hdf5
            # hds = pd.read_csv(self.featfile, index_col=0, nrows=1)
            hds = feat_df.columns
            hds = list(set([hd.split('__')[1] for hd in hds]))

            # option 1, expand rows
            pad_left = int((ti0 - ti) / self.dto)  # if ti < ti0 else 0,
            pad_right = int(((ti + (Nw - 1) * self.dto) - tf0) / self.dto)  # if tf > tf0 else 0
            i0 = abs(pad_left) if pad_left < 0 else 0
            i1 = Nw0 + max([pad_left, 0]) + pad_right

            # option 2, expand columns
            existing_cols = set(hds)  # these features already calculated, in file
            new_cols = set(cfp.keys()) - existing_cols  # these features to be added
            more_cols = bool(new_cols)
            all_cols = existing_cols | new_cols
            cfp = ComprehensiveFCParameters()
            cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in all_cols])

            # option 3, expand both
            # QUESTION what is purpose of update_feature_matrix
            if any([more_cols, pad_left > 0, pad_right > 0]) and self.update_feature_matrix:
                # FIXME need to change to hdf5
                # fm = pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True)
                fm = feat_df[:]
                if more_cols:
                    # expand columns now
                    df0, wd = self._construct_windows(Nw0, ti0)
                    cfp0 = ComprehensiveFCParameters()
                    cfp0 = dict([(k, cfp0[k]) for k in cfp0.keys() if k in new_cols])
                    fm2 = extract_features(df0, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp0,
                                           impute_function=impute)
                    fm2.index = pd.Series(wd)

                    fm = pd.concat([fm, fm2], axis=1, sort=False)

                # check if updates required because training period expanded
                # expanded earlier
                if pad_left > 0:
                    df, wd = self._construct_windows(Nw, ti, i1=pad_left)
                    fm2 = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp,
                                           impute_function=impute)
                    fm2.index = pd.Series(wd)
                    fm = pd.concat([fm2, fm], sort=False)
                    # expanded later
                if pad_right > 0:
                    df, wd = self._construct_windows(Nw, ti, i0=Nw - pad_right)
                    fm2 = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp,
                                           impute_function=impute)
                    fm2.index = pd.Series(wd)
                    fm = pd.concat([fm, fm2], sort=False)

                # write updated file output
                # FIXME need to change to hdf5
                # fm.to_csv(self.featfile, index=True, index_label='time')
                write_to_hdf5(self.featfile, fp, fm)
                # trim output
                fm = fm.iloc[i0:i1]
            else:
                # read relevant part of matrix
                # FIXME need to change to hdf5
                # fm = pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True,
                #                  header=0, skiprows=range(1, i0 + 1), nrows=i1 - i0)
                fm = read_hdh5(self.featfile, fp)
        else:
            # create feature matrix from scratch
            df, wd = self._construct_windows(Nw, ti)
            fm = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp,
                                  impute_function=impute)
            fm.index = pd.Series(wd)
            # FIXME need to change to hdf5
            # fm.to_csv(self.featfile, index=True, index_label='time')
            write_to_hdf5(self.featfile, fp, fm)

        ys = pd.DataFrame(self._get_label(fm.index, eruption), columns=['label'], index=fm.index)
        return fm, ys

    @staticmethod
    def _get_label(ts: pd.Series, eruption: datetime):
        """

        :param ts: List of datetime objects to get the time til eruption
        :param eruption: Eruption datetime to get time til
        :return: Label vector
        """
        label = [0] * len(ts)
        for i in range(len(ts)):
            label[i] = (eruption - ts[i].to_pydatetime()).total_seconds()
        return label

    def _load_data(self, eruptions=None) -> (pd.DataFrame, pd.DataFrame):
        """
        Load feature matrix and target vector
        :return: tuple(fm, ys)
        """
        # # return if already exist
        # if self.fm and self.ys:
        #     return self.fm, self.ys

        if eruptions is None:
            eruptions = self.eps

        # for each eruptive period compute the features
        fm = [None] * len(eruptions)
        ys = [None] * len(eruptions)
        for i, ep in enumerate(eruptions):
            print(f'feature extraction from {ep[0]} to {ep[-1]}')
            fm[i], ys[i] = self._extract_features(ep[0], ep[-1], self.get_erp(ep), feats="comprehensive")

        fm = pd.concat(fm)
        ys = pd.concat(ys)
        self.fm = fm
        self.ys = ys

        return fm, ys

    def get_erp(self, erp_period: List[datetime]):
        for i in self.data.tes:
            if i in erp_period:
                return i

    def read_rel_feats(self, filename: str) -> List[str]:
        with open(filename, "r") as f:
            file_contents = f.readlines()
        feats = [feat.strip() for feat in file_contents[1:]]  # using [1:] to not read the title 'relevant features'
        return feats

    def feature_selection(self, output: bool = True, recompute: bool = False):

        # load the feature matrix and the target label vector
        fm, ys = self._load_data()

        # initial lists to hold information about selected features
        # using 2d arrays, [[],[],[],[],[]] where each inner list will hold 4 lists
        # 1 for each ep_dropped within each eruption not seen
        rel_feats = [[] for i in range(len(self.eps))]
        # p_vals = [[] for i in range(len(self.eps))]
        # all_feats = [[] for i in range(len(self.eps))]

        for idx, erp_not_seen in enumerate(self.eps):  # loop through the eruptive periods
            # exclude the current eruptive period
            inds_not_seen = (fm.index < erp_not_seen[0]) | (erp_not_seen[-1] < fm.index)
            erps_left = self.eps[:idx]+self.eps[idx+1:]

            # get the current eruption
            eruption_not_seen = self.get_erp(erp_not_seen)
            eruption_not_seen = f"{eruption_not_seen.year}-{eruption_not_seen.month}-{eruption_not_seen.day}"
            print(f"selecting features while have not seen {eruption_not_seen}")

            # loop over all other eruptive periods
            for ep in erps_left:
                # get the current eruption
                erp_dropped = self.get_erp(ep)
                erp_dropped = f"{erp_dropped.year}-{erp_dropped.month}-{erp_dropped.day}"
                rel_feat_file = f"{self.featdir}/{eruption_not_seen}/relevant_feats_{erp_dropped}.fts"
                feats_p_value_file = f"{self.featdir}/{eruption_not_seen}/feats_p-values_{erp_dropped}.fts"

                if (not recompute) and os.path.isfile(rel_feat_file):
                    # read in the relevant feature file
                    relevant_features = self.read_rel_feats(rel_feat_file)
                    rel_feats[idx].append(relevant_features)
                    continue  # go to the next iteration of for loop

                # exclude the current eruptive period
                inds = inds_not_seen & ((fm.index < ep[0]) | (ep[-1] < fm.index))
                fmp = fm.loc[inds]
                ysp = ys.loc[inds]

                # find significant features
                select = FeatureSelector(n_jobs=self.n_jobs, ml_task='regression')
                select.fit_transform(fmp, ysp['label'])  # using ['label'] as pd.Series is needed by FeatureSelector

                # remove the spaces in feature names
                def fix_feature_names(name: str):
                    return name.replace(" ", "")
                features = list(map(fix_feature_names, select.features))
                relevant_features = list(map(fix_feature_names, select.relevant_features))

                rel_feats[idx].append(relevant_features)
                # p_vals[idx].append(select.p_values)
                # all_feats[idx].append(features)

                if output:
                    # write features and their p_values to csv
                    makedir(f"{self.featdir}/{eruption_not_seen}")
                    with open(feats_p_value_file, "w") as fp:
                        fp.write("features p_values\n")
                        for feat, p_val in zip(features, select.p_values):
                            fp.write(f"{feat} {p_val}\n")

                    # write relevant features to csv
                    with open(rel_feat_file, "w") as fp:
                        fp.write("relevant_features\n")
                        for feat in relevant_features:
                            fp.write(f"{feat}\n")

        # rel_feats = np.array(rel_feats)
        # union = list(set(chain.from_iterable(rel_feats[0])))  # get the union of all from rel_feats[0]
        return rel_feats


def dt_ceil(dt: datetime):
    secs = dt.hour * 3600 + dt.minute * 60 + dt.second
    if secs == 0:
        # datetime is already exactly at start of day, no need to ceil
        return dt
    return datetime(dt.year, dt.month, dt.day + 1)


def dt_floor(dt: datetime):
    return datetime(dt.year, dt.month, dt.day)
