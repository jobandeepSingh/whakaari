import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel
from obspy import UTCDateTime
from datetime import timedelta, datetime, date

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
    # td = TremorData(raw_data=True, n_jobs=4)
    # print(td.ti)
    # print(td.tf)
    # print(type(td.ti))
    # print(type(td.tf))
    # td.update(tf='2011-01-03')
    # print(td.ti)
    # print(td.tf)
    # print(type(td.ti))
    # print(type(td.tf))

    ##### STANDARD DATA ##########
    # GIFLENS-https://media1.giphy.com/media/xT9IgG50Fb7Mi0prBC/200.gif    

    # td = TremorData(n_jobs=4)
    # print(td.ti)
    # print(td.tf)
    # print(type(td.ti))
    # print(type(td.tf))
    # td.update(tf='2020-03-04')

    # td = TremorData()
    
    # _DAY = timedelta(days=1.)

    # tf = td.tf + _DAY + _DAY
    # t1 = UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(tf.year, tf.month, tf.day))

    # td.update(tf=t1)