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
from typing import List
import tables

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

datas = ['rsam', 'mf', 'hf', 'dsar']
all_classifiers = ["SVM", "KNN", 'DT', 'RF', 'NN', 'NB', 'LR']
MONTH = timedelta(days=365.25 / 12)
_DAY = timedelta(days=1.)
DEFAULT_SECS_BETWEEN_OBS = 1

makedir = lambda name: os.makedirs(name, exist_ok=True)


# class FilePath(object):
#     def __init__(self, date_time: datetime, secs_between_obs: int):
#         self.path = \
#             f"/Year_{date_time.year}/Month_{date_time.month}/Day_{date_time.day}_secs-between-obs={secs_between_obs}"


def get_data_for_day(t0: datetime, store: str, i: int = 0, secs_between_obs: int = DEFAULT_SECS_BETWEEN_OBS):
    """
    Download WIZ data for given 24 hour period, writing data to temporary file.

    :param datetime t0: Initial date of data download period.
    :param str store: Name of hdf5 file to store the raw data.
    :param int i: Number of days that 24 hour download period is offset from initial date.
    :param int secs_between_obs: Desired seconds between observations.
    :return: None
    """
    if failedobspyimport:
        raise ImportError('ObsPy import failed, cannot update data.')

    t0 = UTCDateTime(t0)
    daysec = 24 * 3600
    date_time = (t0 + i * daysec)

    # check if data already in store
    fp = f"/Year_{date_time.year}/Month_{date_time.month}/Day_{date_time.day}_secs-between-obs={secs_between_obs}"
    store = pd.HDFStore(store)
    if fp in store:
        store.close()
        return
    store.close()

    # open clients
    client = FDSNClient("GEONET")
    client_nrt = FDSNClient('https://service-nrt.geonet.org.nz')

    data_streams = [[2, 5], [4.5, 8], [8, 16]]
    names = ['rsam', 'mf', 'hf']

    # download data
    datas = []
    try:
        site = client.get_stations(starttime=t0 + i * daysec, endtime=t0 + (i + 1) * daysec, station='WIZ',
                                   level="response", channel="HHZ")
    except FDSNNoDataException:
        pass

    try:
        WIZ = client.get_waveforms('NZ', 'WIZ', "10", "HHZ", t0 + i * daysec, t0 + (i + 1) * daysec)

        # if less than 1 day of data, try different client
        # QUESTION why less than 600*100
        if len(WIZ.traces[0].data) < 600 * 100:
            raise FDSNNoDataException('')
    except ObsPyMSEEDFilesizeTooSmallError:
        return
    except FDSNNoDataException:
        try:

            WIZ = client_nrt.get_waveforms('NZ', 'WIZ', "10", "HHZ", t0 + i * daysec, t0 + (i + 1) * daysec)
        except FDSNNoDataException:
            return

    # process frequency bands
    WIZ.remove_sensitivity(inventory=site)
    data = WIZ.traces[0].data
    ti = WIZ.traces[0].meta['starttime']
    # round start time to start of day
    tiday = UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day))

    ti = tiday + int(np.round((ti - tiday) / secs_between_obs)) * secs_between_obs
    N = secs_between_obs * 100  # numbers of observations per window in data
    Nm = int(N * np.floor(len(data) / N))
    for data_stream, name in zip(data_streams, names):
        filtered_data = bandpass(data, data_stream[0], data_stream[1], 100)
        filtered_data = abs(filtered_data[:Nm])
        datas.append(filtered_data.reshape(-1, N).mean(axis=-1) * 1.e9)

    # QUESTION Does integration and bandpass order matter? Paper says other way around
    # Would save a lot of below computations if other way around maybe?
    # compute dsar
    data = cumtrapz(data, dx=1. / 100, initial=0)
    data -= np.mean(data)
    j = names.index('mf')
    mfd = bandpass(data, data_streams[j][0], data_streams[j][1], 100)
    mfd = abs(mfd[:Nm])
    mfd = mfd.reshape(-1, N).mean(axis=-1)
    j = names.index('hf')
    hfd = bandpass(data, data_streams[j][0], data_streams[j][1], 100)
    hfd = abs(hfd[:Nm])
    hfd = hfd.reshape(-1, N).mean(axis=-1)
    dsar = mfd / hfd
    datas.append(dsar)
    names.append('dsar')

    # write out file
    datas = np.array(datas)
    time = [(ti + j * secs_between_obs).datetime for j in range(datas.shape[1])]
    df = pd.DataFrame(zip(*datas), columns=names, index=pd.Series(time))
    df.index.name = "DateTime"
    df = df[df.index.day == (t0.day + i)]  # Get rid of rows that aren't actually this day

    store = pd.HDFStore(store)
    store.put(fp, df)
    store.close()


def get_data_between(ti: datetime, tf: datetime, store: str, secs_between_obs: int = None, n_jobs: int = 6) -> None:
    ti = datetimeify(ti)
    tf = datetimeify(tf)
    n_days = (tf - ti).days + 1

    # parallel data collection
    pars = [[ti, store, i] for i in range(n_days)]
    if secs_between_obs:  # add secs_between_obs if given
        for p in pars: p.append(secs_between_obs)

    p = Pool(n_jobs)
    p.starmap(get_data_for_day, pars)
    p.close()
    p.join()


def get_data_for_days(days: List[datetime], store: str, secs_between_obs: int = None, n_jobs: int = 6) -> None:
    # parallel data collection
    pars = [[day, store] for day in days]
    if secs_between_obs:  # add secs_between_obs if given
        for p in pars: p.append(secs_between_obs)

    p = Pool(n_jobs)
    p.starmap(get_data_for_day, pars)
    p.close()
    p.join()


def get_eruptions() -> List[datetime]:
    with open('data/eruptive_periods.txt', 'r') as fp:
        eruptions = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    return eruptions


def datetimeify(t):
    """ Return datetime object corresponding to input string.

        Parameters:
        -----------
        t : str, datetime.datetime
            Date string to convert to datetime object.

        Returns:
        --------
        datetime : datetime.datetime
            Datetime object corresponding to input string.

        Notes:
        ------
        This function tries several datetime string formats, and raises a ValueError if none work.
    """
    if type(t) in [datetime, Timestamp]:
        return t
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S', ]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))


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


if __name__ == "__main__":
    # # ===== USING THE TREMOR DATA CLASS =======
    # t = TremorData()
    # look_forward = 2.
    # delta = timedelta(days=look_forward)
    # for eruption in t.tes:
    #     t.update(ti=eruption-delta, tf=eruption) # does linear interpolation which is not needed
    #     print(f"updated: {eruption-delta} to {eruption}")

    os.chdir('..')  # set working directory to root

    raw_data_store = 'data\\raw_data.h5'
    # put feature h5 file in forecaster class

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
    iw = 60 * 20  # observations per window # 60 obs per min * 20 mins
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
        plt.axvline(x=eruptions[i])
        # add title
        plt.title(f"Eruption: {eruptions[i]}")

        fig.tight_layout()
        plt.show()
