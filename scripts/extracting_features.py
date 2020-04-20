# general imports
import os, sys, shutil, warnings, gc, joblib
import numpy as np
from datetime import datetime, timedelta, date
# from copy import copy
# from matplotlib import pyplot as plt
# from inspect import getfile, currentframe
# from glob import glob
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from multiprocessing import Pool
# from textwrap import wrap
# from time import time
from scipy.integrate import cumtrapz
# from scipy.signal import stft
# from scipy.optimize import curve_fit
# from corner import corner
# from functools import partial
# from fnmatch import fnmatch

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

# feature recognition imports
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.transformers import FeatureSelector
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from imblearn.under_sampling import RandomUnderSampler

# # classifier imports
# from sklearn.metrics import matthews_corrcoef
# from sklearn.model_selection import GridSearchCV, ShuffleSplit
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC

import pyarrow as pa
import pyarrow.parquet as pq

os.chdir('..')

makedir = lambda name: os.makedirs(name, exist_ok=True)

# Changed secs_per_win to secs_between_obs
def get_data_for_day(i,t0,secs_between_obs=1, root="default_data"):
    """ Download WIZ data for given 24 hour period, writing data to temporary file.

        Parameters:
        -----------
        i : integer
            Number of days that 24 hour download period is offset from initial date.
        t0 : datetime.datetime
            Initial date of data download period.
        
    """
    t0 = UTCDateTime(t0)

    # open clients
    client = FDSNClient("GEONET")
    client_nrt = FDSNClient('https://service-nrt.geonet.org.nz')
    
    daysec = 24*3600
    data_streams = [[2, 5], [4.5, 8], [8,16]]
    names = ['rsam','mf','hf']

    # download data
    datas = []
    try:
        site = client.get_stations(starttime=t0+i*daysec, endtime=t0 + (i+1)*daysec, station='WIZ', level="response", channel="HHZ")
    except FDSNNoDataException:
        pass

    try:
        WIZ = client.get_waveforms('NZ','WIZ', "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)
        
        # if less than 1 day of data, try different client
        # TODO why less than 600*100
        if len(WIZ.traces[0].data) < 600*100:
            raise FDSNNoDataException('')
    except ObsPyMSEEDFilesizeTooSmallError:
        return
    except FDSNNoDataException:
        try:
            
            WIZ = client_nrt.get_waveforms('NZ','WIZ', "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)
        except FDSNNoDataException:
            return

    # process frequency bands
    WIZ.remove_sensitivity(inventory=site)
    data = WIZ.traces[0].data
    ti = WIZ.traces[0].meta['starttime']
    # round start time to nearest 10 min increment
    tiday = UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day))

    ti = tiday+int(np.round((ti-tiday)/secs_between_obs))*secs_between_obs
    N = secs_between_obs*100                 # numbers of observations per window in data
    Nm = int(N*np.floor(len(data)/N))
    for data_stream, name in zip(data_streams, names):
        filtered_data = bandpass(data, data_stream[0], data_stream[1], 100)
        filtered_data = abs(filtered_data[:Nm])
        datas.append(filtered_data.reshape(-1,N).mean(axis=-1)*1.e9)

    
    # QUESTION Does integration and bandpass order matter? Paper says other way around
    # Would save a lot of below computations if other way around maybe?
    # compute dsar
    data = cumtrapz(data, dx=1./100, initial=0)
    data -= np.mean(data)
    j = names.index('mf')
    mfd = bandpass(data, data_streams[j][0], data_streams[j][1], 100)
    mfd = abs(mfd[:Nm])
    mfd = mfd.reshape(-1,N).mean(axis=-1)
    j = names.index('hf')
    hfd = bandpass(data, data_streams[j][0], data_streams[j][1], 100)
    hfd = abs(hfd[:Nm])
    hfd = hfd.reshape(-1,N).mean(axis=-1)
    dsar = mfd/hfd
    datas.append(dsar)
    names.append('dsar')

    # write out file
    datas = np.array(datas)
    time = [(ti+j*secs_between_obs).datetime for j in range(datas.shape[1])]
    df = pd.DataFrame(zip(*datas), columns=names, index=pd.Series(time))
    df.index.name = "DateTime"
    date_time = (t0+i*daysec)
    df = df[df.index.day == (t0.day+i)] # Get rid of rows that aren't actually this day
    fp =f"{root}/{date_time.year}/{date_time.month}"
    table = pa.Table.from_pandas(df, preserve_index=False)
    makedir(fp)
    # df.to_csv(f"{fp}/{date_time.day}.csv", index=True, index_label='time')
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f'{fp}/{date_time.day}.parquet')


def read_dfs(root, date_times):
    """
        inputs:
            root - string for root of dataframe
            date_times - list of date time objects for dates to read in
        
        returns:
            dataframe with the specified dates
    """
    # TODO secs_between_obs parameter and downsample the 1 second raw data to get that
    dfs = []
    for date_time in date_times:
        file_path = f"{root}/{date_time.year}/{date_time.month}/{date_time.day}.parquet"
        if os.path.isfile(file_path):
            dfs.append(pq.read_pandas(file_path).to_pandas())
        else:
            raise ValueError(f"File does not exist: {file_path} \n {os.getcwd()}")

    return dfs


def get_data(root, days, n_jobs, secs_between_obs=1):
    # parallel data collection
    pars = [[0,day,secs_between_obs, root] for day in days]

    p = Pool(n_jobs)
    p.starmap(get_data_for_day, pars)
    p.close()
    p.join()


# TODO use secs_per_window, pass in date instead of df
def feature_extraction(df, root, window_overlap, obs_per_window, secs_between_obs, n_jobs, recompute = False, 
                        source_win_overlap=None, source_obs_per_win=None, source_secs_between_obs=None):
    iw = obs_per_window
    io = int(iw * window_overlap)
    # dto - length of non-overlapping section of window (timedelta)
    dto = (1-window_overlap) * timedelta(seconds = secs_between_obs*iw)
        
    # TODO this should work with any time between observations right?
    # number of windows in feature request
    Nw = int(df.shape[0]/(iw-io))
    
    cfp = MinimalFCParameters()
    # cfp = ComprehensiveFCParameters()
    # if self.compute_only_features:
    #     cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in self.compute_only_features])
    # else:
    #     # drop features if relevant
    #     _ = [cfp.pop(df) for df in self.drop_features if df in list(cfp.keys())]

    date_time = df.index[0]

    # Transpose the df before saving if performance is rubbish
    # TODO maybe change days to folders
    if source_win_overlap and source_obs_per_win:
        original_file_path = f"{root}/{date_time.year}/{date_time.month}/{date_time.day}_overlap={source_win_overlap}_obs-per-window={source_obs_per_win}_secs-between-obs={source_secs_between_obs}_features"
            
        if not os.path.isfile(f"{original_file_path}.parquet"):
            raise FileNotFoundError("You goofed up, better luck next time!")

        file_path = f"{original_file_path}_overlap={window_overlap}_obs-per-window={window_overlap}_secs-between-obs={secs_between_obs}_meta_features.parquet"
                
    else:
        file_path = f"{root}/{date_time.year}/{date_time.month}/{date_time.day}_overlap={window_overlap}_obs-per-window={obs_per_window}_secs-between-obs={secs_between_obs}_features.parquet"

    
    if os.path.isfile(file_path) and not recompute:
        fm = pq.read_pandas(file_path).to_pandas()
        if source_win_overlap and source_obs_per_win:
            fm = fm.transpose()
        return fm
    else:
        # create feature matrix from scratch   
        df, wd = construct_windows(df, Nw, iw, io, dto)
        # fm = extract_features(df, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=impute)
        fm = extract_features(df, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=None)
        fm.index = pd.Series(wd)
        fm.index.name = "DateTime"
        # fm = fm.loc[~(fm == 0).all(axis=1)]
        # Remove any cols that are all NaN
        fm = fm.iloc[:,~fm.isnull().values.all(axis=0)]
        # Impute any NaN values left
        impute(fm)
        # Get rid of 'if' if it doesn't take too long
        if not (source_win_overlap and source_obs_per_win):
            fm.columns = [name.replace("_", "-") for name in fm.columns]
            table = pa.Table.from_pandas(fm)
            pq.write_table(table, file_path)
        else:
            fm = fm.transpose()
            table = pa.Table.from_pandas(fm)
            pq.write_table(table, file_path)
            fm = fm.transpose()

    return fm


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
    window_dates = []
    diff = df.index[1] - df.index[0]
    for i in range(Nw):
        dfi = df[:].iloc[i*(iw-io):i*(iw-io)+iw]
        dfi['id'] = i+1
        dfs.append(dfi)
        window_dates.append(ti+ i*dto) # timedelta(i*(iw-io))
    df = pd.concat(dfs)
    # window_dates = [ti + i*self.dto for i in range(Nw)]
    return df, window_dates


def feature_selection(root, df, days_ahead=1, n_jobs=4):
    with open('data/eruptive_periods.txt','r') as fp:
        tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

    def temp(from_time):
        for te in tes:
            if 0 < (te-from_time).total_seconds()/(3600*24) < days_ahead:
                return 1
        return 0

    krakatoa = pd.Series(list(map(temp, df.index)), index=df.index)
    # krakatoa = pd.Series(np.array([0]*16 + [1]*16), index=fm.index)

    select = FeatureSelector(n_jobs=n_jobs, ml_task='classification')
    select.fit_transform(df, krakatoa)
    print('Finish select features')

    Nfts = 100
    fts = select.features[:Nfts]
    pvs = select.p_values[:Nfts]
    date_time = df.index[0]
    with open(f"{root}/{df.index[0].date()}_{df.index[-1].date()}_davids_features.txt",'w') as fp:
        fp.write('features p_values\n')
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))
    print('Finish david file')

    fts = select.relevant_features[:Nfts]
    pvs = select.feature_importances_[:Nfts]
    with open(f"{root}/{df.index[0].date()}_{df.index[-1].date()}_relevant_features.txt",'w') as fp:
        fp.write('relevant_features feature_importances_\n')
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))
    print('Finish other file')


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
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S',]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))


if __name__ == "__main__":
    days = [datetime(2011,1,2), datetime(2012,7,5), datetime(2020,4,18), datetime(2019,12,8)]
    # get_data('Small', days, n_jobs=4)
    raw_data_list = read_dfs('Small', days)
    source_df = feature_extraction(raw_data_list[1], 'Small', 0.5, 20, 1, 4)
    meta_df = feature_extraction(source_df, 'Small', 0.5, 120, 10, 4, source_win_overlap=0.5, source_obs_per_win=20, source_secs_between_obs=1)

    source_df1 = feature_extraction(raw_data_list[3], 'Small', 0.5, 20, 1, 4)
    meta_df1 = feature_extraction(source_df1, 'Small', 0.5, 120, 10, 4, source_win_overlap=0.5, source_obs_per_win=20, source_secs_between_obs=1)
    
    print('starting feature selection')
    new_df = pd.concat([meta_df,meta_df1])
    feature_selection("Small",new_df, 1, 4)

    print('hello')
