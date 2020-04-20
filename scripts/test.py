import os, sys
#sys.path.insert(0, os.path.abspath('.'))
#from whakaari import TremorData, ForecastModel
from obspy import UTCDateTime
from datetime import timedelta, datetime, date
import pandas as pd
from inspect import getfile, currentframe
    
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
    #t0 = UTCDateTime("2011-01-01 00:00:00")
    #t1 = UTCDateTime("2011-01-02 00:00:00")
    #print(type(t0))
    df = pd.read_csv('tremor_data.dat', index_col=0, parse_dates=[0,], infer_datetime_format=True)
    print(type(df.index[0]))

