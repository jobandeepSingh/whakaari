import os, sys
#sys.path.insert(0, os.path.abspath('.'))
#from whakaari import TremorData, ForecastModel
from obspy import UTCDateTime
from datetime import timedelta, datetime, date
import pandas as pd
from inspect import getfile, currentframe
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.transformers import FeatureSelector
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
import numpy as np


# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


if __name__ == "__main__":
    print("reading df1 and df2")
    df1 = pd.read_csv('features_tremor_data.dat', index_col=0, parse_dates=[0,], infer_datetime_format=True)
    df2 = pd.read_csv('features_tremor_data(1).dat', index_col=0, parse_dates=[0,], infer_datetime_format=True)
    # GIFLENS-https://media1.giphy.com/media/3oxOCmLMY1XUa9LqlG/200.gif    
    # df3 = pd.concat([df1,df2])
    # df3.to_csv("feature_data.dat", index=True)
    # df1['krakatoa'] = 0
    # df2['krakatoa'] = 1

    # df1.columns = [name.replace("_", "-") for name in df1.columns]
    # df2.columns = [name.replace("_", "-") for name in df2.columns]


    df1_sample = df1.iloc[::100, :]
    df2_sample = df2.iloc[::100, :]
    print('Finished sampling')
    # dfs1 = []
    # dfs2 = []

    Nw = int(np.floor(87/5))-1
    i1 = Nw
    i0 = 0
    window_dates_1 = []
    window_dates_2 = []

    for i in range(i0, i1):
    #     dfi1 = df1_sample[:].iloc[i*5:i*5+10]
    #     dfi2 = df2_sample[:].iloc[i*5:i*5+10]
        window_dates_1.append(df1_sample.index[i*5])
        window_dates_2.append(df2_sample.index[i*5])
    #     try:
    #         dfi1['id'] = i # pd.Series(np.ones(self.iw, dtype=int)*i, index=dfi.index)
    #         dfi2['id'] = i
    #     except ValueError:
    #         print('KRAKATOA!!')
    #     dfs1.append(dfi1)
    #     dfs2.append(dfi2)

    # print('Finished creating windows')
    # df1_windows = pd.concat(dfs1)
    # df2_windows = pd.concat(dfs2)
    # window_dates_1 =  df1.iloc[::100, :][:-2].index
    # window_dates_2 =  df2.iloc[::100, :][:-2].index

    # extract freatures
    # print('Start feature extract')
    # cfp = ComprehensiveFCParameters()
    # fm1 = extract_features(df1_windows, column_id='id', n_jobs=6, default_fc_parameters=cfp, impute_function=impute)
    # fm1.to_csv("fm1.dat", index=True)
    # print('Start feature extract2')
    # fm2 = extract_features(df2_windows, column_id='id', n_jobs=6, default_fc_parameters=cfp, impute_function=impute)
    # fm2.to_csv("fm2.dat", index=True)
    # print('Finish feature extract2')
    del(df1)
    del(df2)
    print("reading fm1 and fm2")
    fm1 = pd.read_csv("fm1.dat", index_col=0, parse_dates=[0,], infer_datetime_format=True)
    fm2 = pd.read_csv("fm2.dat", index_col=0, parse_dates=[0,], infer_datetime_format=True)
    print(fm1.shape)
    print(fm2.shape)
    fm1.index = window_dates_1
    fm2.index = window_dates_2
    fm = pd.concat([fm1, fm2])
    krakatoa = np.array([0]*85 + [1]*85)
    fm.to_csv("feature_matrix.dat", index=True)
    print('Finish writing feature matrix to csv')

    

    select = FeatureSelector(n_jobs=6, ml_task='classification')
    select.fit_transform(fm, krakatoa)
    print('Finish select features')

    Nfts = 100
    fts = select.features[:Nfts]
    pvs = select.p_values[:Nfts]
    with open('davids_features.txt','w') as fp:
        fp.write('features p_values\n')
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))
    print('Finish david file')

    fts = select.relevant_features[:Nfts]
    pvs = select.feature_importances_[:Nfts]
    with open('relevant_features.txt','w') as fp:
        fp.write('relevant_features feature_importances_\n')
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))
    print('Finish other file')
