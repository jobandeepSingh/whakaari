# general imports
import os, sys, shutil, warnings, gc, joblib
import numpy as np
from datetime import datetime, timedelta, date
import pandas as pd
from multiprocessing import Pool
from scipy.integrate import cumtrapz


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


import pyarrow as pa
import pyarrow.parquet as pq

makedir = lambda name: os.makedirs(name, exist_ok=True)

def get_data_for_day(i,t0,secs_per_win=600, root="default_data"):
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

    ti = tiday+int(np.round((ti-tiday)/secs_per_win))*secs_per_win
    N = secs_per_win*100                 # numbers of observations per window in data
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
    time = [(ti+j*secs_per_win).datetime for j in range(datas.shape[1])]
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


def read_df(root, date_times):
    """
    inputs:
        root - string for root of dataframe
        date_times - list of date time objects for dates to read in
    
    returns:
        dataframe with the specified dates
    """
    dfs = []
    for date_time in date_times:
        file_path = f"{root}/{date_time.year}/{date_time.month}/{date_time.day}.parquet"
        if os.path.exists(file_path):
            dfs.append(pq.read_pandas(file_path).to_pandas())
        else:
            raise ValueError(f"File does not exist: {file_path}")

    return pd.concat(dfs)


def get_data(root, days, n_jobs):
    # parallel data collection
    secs_between_obs = 1
    pars = [[0,day,secs_between_obs, root] for day in days]

    p = Pool(n_jobs)
    p.starmap(get_data_for_day, pars)
    p.close()
    p.join()


if __name__ == "__main__":
    days = [datetime(2011,1,2), datetime(2012,7,5), datetime(2020,4,18)]
    get_data('Testing', days, n_jobs=4)

    # TODO dsar artefact removal

    # TODO maybe don't redownload for days that already exist?