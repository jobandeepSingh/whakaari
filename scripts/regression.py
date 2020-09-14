import os
import sys
import warnings

sys.path.insert(0, os.path.abspath('..'))
from datetime import timedelta, datetime
from pandas._libs.tslibs.timestamps import Timestamp
from multiprocessing import Pool
from scipy.integrate import cumtrapz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List
import tables
from statistics import median
from inspect import getfile, currentframe
from scripts.TremorData import DEFAULT_SECS_BETWEEN_OBS, get_data_for_days, datetimeify, TremorData, get_data_for_day
from scripts.RegressionModel import RegressionModel

# ignoring warnings related to naming convention of hdf5
warnings.simplefilter('ignore', tables.NaturalNameWarning)

# ObsPy imports
try:
    from obspy.clients.fdsn import Client as FDSNClient
    from obspy import UTCDateTime, read_inventory

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    from obspy.signal.filter import bandpass
    from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError
    from obspy.clients.fdsn.header import FDSNNoDataException

    failedobspyimport = False
except:
    failedobspyimport = True


def get_eruptions() -> List[datetime]:
    with open('data/eruptive_periods.txt', 'r') as fp:
        eruptions = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    return eruptions


def get_days_either_side(days: List[datetime], days_either_side: int) -> List[datetime]:
    delta = timedelta(days=days_either_side)
    all_days = []
    for day in days:
        eruption_day = datetime(day.year, day.month, day.day)
        for i in range(days_either_side * 2 + 1):
            all_days.append(eruption_day - delta + timedelta(days=i))
    return all_days


def read_data(store_path: str, days: List[datetime], secs_between_obs=DEFAULT_SECS_BETWEEN_OBS) -> List[pd.DataFrame]:
    store = pd.HDFStore(store_path)
    data = [None] * len(days)
    for i in range(len(days)):
        fp = f"/Year_{days[i].year}/Month_{days[i].month}/Day_{days[i].day}_secs-between-obs={secs_between_obs}"
        data[i] = store[fp]
    store.close()
    return data


def construct_binary_response(data: pd.DataFrame, days_forward: float = 2.) -> List[float]:
    eruptions = get_eruptions()
    return [classify_before(date, eruptions, days_forward) for date in data.index]


def classify_before(from_time: datetime, eruptions: List[datetime], days_forward: float = 2.) -> float:
    for eruption in eruptions:
        if 0 < (eruption - from_time).total_seconds() / (3600 * 24) < days_forward:
            return 1.
    return 0.


def classify(from_time: datetime, eruptions: List[datetime], days: float = 2.) -> float:
    for eruption in eruptions:
        if abs((eruption - from_time).total_seconds() / (3600 * 24)) < days:
            return 1.
    return 0.


def construct_windows(df, Nw, iw, io, dto):
    """ Create overlapping data windows for feature extraction.

        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe of data to contruct window.
        Nw : int
            Number of windows to create.
        iw : int
            Number of observations per window.
        io : int
            Number of overlapping observations between window.

        Returns:
        --------
        df : pandas.DataFrame
            Dataframe of windowed data, with 'id' column denoting individual windows.
        window_dates : list
            Datetime objects corresponding to the beginning of each data window.
    """
    ti = df.index[0]
    dfs = []
    window_dates = [None] * Nw

    for i in range(Nw):
        dfi = df[:].iloc[i * (iw - io):i * (iw - io) + iw]
        dfi['id'] = i + 1
        dfs.append(dfi)
        window_dates[i] = ti + i * dto
    df = pd.concat(dfs)
    return df, window_dates


def create_plots_regression():
    """
    Creates the plot for each eruption showing:
        - binary target vector
        - continuous regression target vector
        - eruption date
        - rsam signal

    :return: None
    """

    raw_data_store = 'data\\raw_data.h5'

    # get eruption dates
    eruptions = get_eruptions()
    # get list of days around eruptions
    days = get_days_either_side(eruptions, 2)
    # get data for the days around eruptions
    get_data_for_days(days, raw_data_store)

    # read in data
    raw_data = read_data(raw_data_store, days)
    raw_data_df = pd.concat(raw_data)
    # # get the binary classification vector
    # bin_res = construct_binary_response(raw_data_df)

    # parameters for windows
    iw = 60 * 20  # observations per window : 60 obs per min * 20 mins
    overlap = 0.5  # overlap percentage
    io = int(iw * overlap)  # observations in overlapping part of window
    # length of non-overlapping section of window
    dto = (1 - overlap) * timedelta(seconds=DEFAULT_SECS_BETWEEN_OBS * iw)

    # construct windows for each day
    data_in_win = []
    win_datetimes = []
    for data in raw_data:
        Nw = int(data.shape[0] / (iw - io))
        data_win, win_dt = construct_windows(data, Nw, iw, io, dto)
        data_in_win.append(data_win)
        win_datetimes.append(win_dt)

    # get time until eruption for 48 hours before eruption
    time_to_eruption = []
    for i in range(len(win_datetimes)):
        diff_erp = []
        eruption = eruptions[i // len(eruptions)]  # get the corresponding eruption
        for dt in win_datetimes[i]:
            if classify(dt, [eruption]):
                diff_erp.append((eruption - dt).total_seconds() / 60 / 60)
            else:
                diff_erp.append(0)
        time_to_eruption.append(diff_erp)

    # ========== PLOTTING CODE BELOW ===============
    for i in range(len(eruptions)):
        fig, ax1 = plt.subplots()
        color = 'k'
        ax1.set_xlabel("Time")
        ax1.set_ylabel("RSAM SIGNAL", color=color)
        ax1.tick_params(axis='y', color=color)
        # plot the raw data
        for j in range(i * 5, i * 5 + 5):
            # taking every 600th point so essentially plotting every 10min data
            section = raw_data[j][::600].loc[:, 'rsam']
            ax1.plot(section.index.values, section.values, c=color)
        _, ymax = ax1.get_ylim()
        ax1.set_ylim(bottom=-1. * ymax)  # hacky way to get the y-axis to line up at 0
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        old_color = color
        color = 'r'
        ax2.set_ylabel("Time until Eruption (Hours)", color=color)
        ax2.spines['left'].set_color(old_color)
        ax2.spines['right'].set_color(color)
        ax2.tick_params(axis='y', color=color)
        for j in range(i * 5, i * 5 + 5):
            # plot time to eruptions (from 2 days before eruption up to eruption)
            plt.plot(win_datetimes[j], time_to_eruption[j], c=color)

            # plot the binary response scaled up
            section = raw_data[j][::300]
            bin_res = construct_binary_response(section)
            bin_res = np.array(bin_res)
            bin_res = bin_res * 10
            plt.plot(section.index, bin_res, c='g')

        # plot vertical line for eruption
        plt.axvline(x=eruptions[i], color='b')
        # add title
        plt.title(f"Eruption: {eruptions[i]}")
        # legend
        lines = [Line2D([0], [0], color='k', label='rsam'),
                 Line2D([0], [0], color='g', label='binary target vector'),
                 Line2D([0], [0], color='r', label='regression target vector'),
                 Line2D([0], [0], color='b', label='eruption')]
        plt.legend(handles=lines, loc='lower left')
        fig.tight_layout()
        plt.show()


# put feature h5 file in forecaster class
if __name__ == "__main__":
    os.chdir('..')  # set working directory to root

    # create_plots_regression()

    # td = TremorData()
    #
    # # get_data_for_day(datetime(2012, 8, 3), td.file, 0)
    #
    # days_bracket = 2
    # delta = timedelta(days=days_bracket)
    # for erp in td.tes:
    #     td.update(erp-delta, erp+delta)

    # rm = RegressionModel(window=30, period_before=48)  # NORMAL
    # rm = RegressionModel(window=30, period_before=48, freg=True)  # NORMAL, with F_regression selection
    # rm = RegressionModel(window=30, period_before=48,
    #                       data_streams=['rsam', 'mf', 'hf', 'dsar', 'inv_rsam'])  # INVERSE
    rm = RegressionModel(window=30, period_before=48,  # INVERSE, with F_regression selection
                          data_streams=['rsam', 'mf', 'hf', 'dsar', 'inv_rsam'], freg=True)
    features = rm.feature_selection()
    aggregated_features_4 = rm.feature_aggregation(features, 4)
    aggregated_features_3 = rm.feature_aggregation(features, 3)
    aggregated_features_2 = rm.feature_aggregation(features, 2)
    aggregated_features_1 = rm.feature_aggregation(features, 1)
    rm.update_fm_col_names()
    rm.train(aggregated_features_4, 'LR', '4')
    rm.train(aggregated_features_3, 'LR', '3')
    rm.train(aggregated_features_2, 'LR', '2')
    rm.train(aggregated_features_1, 'LR', '1')

    rm.train(aggregated_features_4, 'RF', '4')
    rm.train(aggregated_features_3, 'RF', '3')
    rm.train(aggregated_features_2, 'RF', '2')
    rm.train(aggregated_features_1, 'RF', '1')

    rm.train(aggregated_features_4, 'GBR', '4')
    rm.train(aggregated_features_3, 'GBR', '3')
    rm.train(aggregated_features_2, 'GBR', '2')
    rm.train(aggregated_features_1, 'GBR', '1')


    print("end")

    # # ====== Updating some data as its patchy =====
    # store = os.sep.join(getfile(currentframe()).split(os.sep)[:-2] + ['data', 'raw_data.h5'])
    # dates = [datetime(2012, 8, 7), datetime(2013, 8, 22), datetime(2013, 10, 6),
    #          datetime(2016, 4, 30), datetime(2019, 12, 12)]
    # for d in dates:
    #     print(f"Getting data for {d}")
    #     get_data_for_day(d, store, overwrite=True)
    # print("Done updating data")
