import os, sys, shutil, warnings

sys.path.insert(0, os.path.abspath('..'))
from datetime import timedelta, datetime
from pandas._libs.tslibs.timestamps import Timestamp
from multiprocessing import Pool
from scipy.integrate import cumtrapz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.signal import stft
from inspect import getfile, currentframe
# from scripts.regression import get_data_for_day
import tables
from statistics import median

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
all_classifiers = ["SVM","KNN",'DT','RF','NN','NB','LR']
MONTH = timedelta(days=365.25 / 12)
_DAY = timedelta(days=1.)
DEFAULT_SECS_BETWEEN_OBS = 1


def makedir(name: str):
    os.makedirs(name, exist_ok=True)


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


class TremorData(object):
    """ Object to manage acquisition and processing of seismic data.

        Attributes:
        -----------
        df : pandas.DataFrame
            Time series of tremor data and transforms.
        t0 : datetime.datetime
            Beginning of data range.
        t1 : datetime.datetime
            End of data range.
        Methods:
        --------
        update
            Obtain latest GeoNet data.
        get_data
            Return tremor data in requested date range.
        plot
            Plot tremor data.
    """

    def __init__(self, n_jobs=6):
        self.n_jobs = n_jobs
        self.file = os.sep.join(getfile(currentframe()).split(os.sep)[:-2] + ['data', 'raw_data.h5'])
        self._assess()

    def __repr__(self):
        if self.exists:
            tm = [self.ti.year, self.ti.month, self.ti.day, self.ti.hour, self.ti.minute]
            tm += [self.tf.year, self.tf.month, self.tf.day, self.tf.hour, self.tf.minute]
            return 'TremorData:{:d}/{:02d}/{:02d} {:02d}:{:02d} to {:d}/{:02d}/{:02d} {:02d}:{:02d}'.format(*tm)
        else:
            return 'no data'

    def _assess(self):
        """ Load existing file and check date range of data.
        """
        # get eruptions
        with open(os.sep.join(getfile(currentframe()).split(os.sep)[:-2] + ['data', 'eruptive_periods.txt']),
                  'r') as fp:
            self.tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        # check if data file exists
        # TODO Better Checks
        self.exists = os.path.isfile(self.file)
        if not self.exists:
            t0 = datetime(2011, 1, 1)
            t1 = datetime(2011, 1, 2)
            self.update(t0, t1)

        # read in the data
        self.df = read_hdh5(self.file)
        self.ti = self.df.index[0]
        self.tf = self.df.index[-1]

    def _compute_transforms(self):
        """ Compute data transforms.
            Notes:
            ------
            Naming convention is *transform_type*_*data_type*, so for example
            'inv_mf' is "inverse medium frequency or 1/mf. Other transforms are
            'diff' (derivative), 'log' (base 10 logarithm) and 'stft' (short-time
            Fourier transform averaged across 40-45 periods).
        """
        for col in self.df.columns:
            if col is 'time': continue
            # inverse
            if 'inv_' + col not in self.df.columns:
                self.df['inv_' + col] = 1. / self.df[col]
            # diff
            if 'diff_' + col not in self.df.columns:
                self.df['diff_' + col] = self.df[col].diff()
                self.df['diff_' + col][0] = 0.
            # log
            if 'log_' + col not in self.df.columns:
                self.df['log_' + col] = np.log10(self.df[col])
            # stft
            if 'stft_' + col not in self.df.columns:
                seg, freq = [12, 16]
                data = pd.Series(np.zeros(seg * 6 - 1))
                data = data.append(self.df[col], ignore_index=True)
                Z = abs(stft(data.values, window='nuttall', nperseg=seg * 6, noverlap=seg * 6 - 1, boundary=None)[2])
                self.df['stft_' + col] = np.mean(Z[freq:freq + 2, :], axis=0)

    def _is_eruption_in(self, days, from_time):  # FIXME add another parameter to look backward too?
        """ Binary classification of eruption imminence.
            Parameters:
            -----------
            days : float
                Length of look-forward.
            from_time : datetime.datetime
                Beginning of look-forward period.
            Returns:
            --------
            label : int
                1 if eruption occurs in look-forward, 0 otherwise

        """
        for te in self.tes:
            if 0 < (te - from_time).total_seconds() / (3600 * 24) < days:
                return 1.
        return 0.

    def update(self, ti=None, tf=None):
        """ Obtain latest GeoNet data.
            Parameters:
            -----------
            ti : str, datetime.datetime
                First date to retrieve data (default is first date data available).
            tf : str, datetime.datetime
                Last date to retrieve data (default is current date).
        """
        if failedobspyimport:
            raise ImportError('ObsPy import failed, cannot update data.')

        makedir('_tmp')

        # default data range if not given
        ti = ti or datetime(self.tf.year, self.tf.month, self.tf.day, 0, 0, 0)
        tf = tf or datetime.today() + _DAY

        ti = datetimeify(ti)
        tf = datetimeify(tf)

        # adding 1 as the difference between tf and ti does not count tf
        ndays = (tf - ti).days + 1

        # parallel data collection
        # pars = [[i, ti] for i in range(ndays)]
        pars = [[ti, self.file, i] for i in range(ndays)]
        p = Pool(self.n_jobs)
        p.starmap(get_data_for_day, pars)
        p.close()
        p.join()

        self.df = read_hdh5(self.file)

        # # impute missing data using linear interpolation and save file
        # self.df = self.df.loc[~self.df.index.duplicated(keep='last')]
        # # self.df = self.df.resample('10T').interpolate('linear')
        #
        # # remove artefact in computing dsar
        # for i in range(1, int(np.floor(self.df.shape[0] / (24 * 6)))):
        #     ind = i * 24 * 6
        #     self.df['dsar'][ind] = 0.5 * (self.df['dsar'][ind - 1] + self.df['dsar'][ind + 1])
        # self.df.to_csv(self.file, index=True)

        self.ti = self.df.index[0]
        self.tf = self.df.index[-1]

    def get_data(self, ti=None, tf=None):
        """ Return tremor data in requested date range.
            Parameters:
            -----------
            ti : str, datetime.datetime
                Date of first data point (default is earliest data).
            tf : str, datetime.datetime
                Date of final data point (default is latest data).
            Returns:
            --------
            df : pandas.DataFrame
                Data object truncated to requsted date range.
        """
        # set date range defaults
        if ti is None:
            ti = self.ti
        if tf is None:
            tf = self.tf

        # convert datetime format
        ti = datetimeify(ti)
        tf = datetimeify(tf)

        # subset data
        inds = (self.df.index >= ti) & (self.df.index < tf)
        return self.df.loc[inds]

    def plot(self, data_streams='rsam', save='tremor_data.png', ylim=[0, 5000]):
        """ Plot tremor data.
            Parameters:
            -----------
            save : str
                Name of file to save output.
            data_streams : str, list
                String or list of strings indicating which data or transforms to plot (see below).
            ylim : list
                Two-element list indicating y-axis limits for plotting.

            data type options:
            ------------------
            rsam - 2 to 5 Hz (Real-time Seismic-Amplitude Measurement)
            mf - 4.5 to 8 Hz (medium frequency)
            hf - 8 to 16 Hz (high frequency)
            dsar - ratio of mf to hf, rolling median over 180 days
            transform options:
            ------------------
            inv - inverse, i.e., 1/
            diff - finite difference derivative
            log - base 10 logarithm
            stft - short-time Fourier transform at 40-45 min period
            Example:
            --------
            data_streams = ['dsar', 'diff_hf'] will plot the DSAR signal and the derivative of the HF signal.
        """
        if type(data_streams) is str:
            data_streams = [data_streams, ]
        if any(['_' in ds for ds in data_streams]):
            self._compute_transforms()

        # set up figures and axes
        f = plt.figure(figsize=(24, 15))
        N = 10
        dy1, dy2 = 0.05, 0.05
        dy3 = (1. - dy1 - (N // 2) * dy2) / (N // 2)
        dx1, dx2 = 0.43, 0.03
        axs = [plt.axes([0.05 + (1 - i // (N / 2)) * (dx1 + dx2), dy1 + (i % (N / 2)) * (dy2 + dy3), dx1, dy3]) for i in
               range(N)][::-1]

        for i, ax in enumerate(axs):
            ti, tf = [datetime.strptime('{:d}-01-01 00:00:00'.format(2011 + i), '%Y-%m-%d %H:%M:%S'),
                      datetime.strptime('{:d}-01-01 00:00:00'.format(2012 + i), '%Y-%m-%d %H:%M:%S')]
            ax.set_xlim([ti, tf])
            ax.text(0.01, 0.95, '{:4d}'.format(2011 + i), transform=ax.transAxes, va='top', ha='left', size=16)
            ax.set_ylim(ylim)

        # plot data for each year
        data = self.get_data()
        xi = datetime(year=1, month=1, day=1, hour=0, minute=0, second=0)
        cols = ['c', 'm', 'y', 'g', [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]
        for i, ax in enumerate(axs):
            if i // (N / 2) == 0:
                ax.set_ylabel('data [nm/s]')
            else:
                ax.set_yticklabels([])
            x0, x1 = [xi + timedelta(days=xl) - _DAY for xl in ax.get_xlim()]
            inds = (data.index >= x0) & (data.index <= x1)
            for data_stream, col in zip(data_streams, cols):
                ax.plot(data.index[inds], data[data_stream].loc[inds], '-', color=col, label=data_stream)

            for te in self.tes:
                ax.axvline(te, color='k', linestyle='--', linewidth=2)
            ax.axvline(te, color='k', linestyle='--', linewidth=2, label='eruption')
        axs[-1].legend()

        plt.savefig(save, dpi=400)


def read_hdh5(store_path: str):
    store = pd.HDFStore(store_path)
    data = []
    for key in store.keys():
        data.append(store[key])
    store.close()
    data = pd.concat(data)
    data.sort_index(inplace=True)
    return data


def write_to_hdf5(store_path: str, fp: str, df: pd.DataFrame):
    store = pd.HDFStore(store_path)
    store.put(fp, df)
    store.close()


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

    datas = np.array(datas)
    time = [(ti + j * secs_between_obs).datetime for j in range(datas.shape[1])]
    df = pd.DataFrame(zip(*datas), columns=names, index=pd.Series(time))
    df.index.name = "DateTime"
    df = df[df.index.day == (t0.day + i)]  # Get rid of rows that aren't actually this day

    # QUESTION do we need to remove duplicates if we already trim data to only that day?
    # remove duplicate indicies (LEGACY code from David)
    df = df.loc[~df.index.duplicated(keep='last')]

    # remove artefact in computing dsar
    median_dsar = median(df['dsar'])
    for i in range(5):  # only show up in first 5 seconds of data
        df['dsar'][i] = median_dsar

    # write out file
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
        for p in pars:
            p.append(secs_between_obs)

    p = Pool(n_jobs)
    p.starmap(get_data_for_day, pars)
    p.close()
    p.join()
