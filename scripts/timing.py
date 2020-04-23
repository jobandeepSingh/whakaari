import os
import numpy as np
from time import time
from datetime import datetime
from shutil import copyfile
from extracting_features_hdf5 import get_data, read_dfs, feature_extraction, console_print

if __name__ == "__main__":

    if os.path.isfile('D:\\0 2020 sem 1\\whakaari\\dataset_test.h5'):
        os.remove('D:\\0 2020 sem 1\\whakaari\\dataset_test.h5')

    num_times = 2
    n_jobs = 6

    # # timing the full process of:
    # #   - downloading data for 1 day
    # #   - extracting features
    # #   - extracting meta features
    # console_print("Starting Timing FULL")
    # times_full = []
    # for i in range(num_times):
    #     # add randomisation of date
    #     start = time()
    #     os.system("python extracting_features_hdf5.py 2012 5 18")
    #     end = time()
    #     print()
    #     os.remove('D:\\0 2020 sem 1\\whakaari\\dataset_test.h5')
    #     times_full.append(end-start)
    
    # with open(f'D:\\0 2020 sem 1\\whakaari\\times_full_{num_times}.txt', 'w') as f:
    #     for i in times_full:
    #         f.write(f"{i}\n")

    os.chdir('D:\\0 2020 sem 1\\whakaari')

    
    # # timing of:
    # #   - downloading data for 1 day
    # console_print("Starting Timing Data Download")
    # times_dowload = []
    # for i in range(num_times):
    #     start = time()
    #     days = [datetime(2011,5,2)]
    #     store = f'dataset_test.h5'
    #     get_data(store, days, n_jobs)
    #     raw_data_list = read_dfs(store, days)

    #     end = time()
    #     os.remove(f'D:\\0 2020 sem 1\\whakaari\\{store}')
    #     times_dowload.append(end-start)
    
    # with open(f'D:\\0 2020 sem 1\\whakaari\\times_dowload_{num_times}.txt', 'w') as f:
    #     for i in times_dowload:
    #         f.write(f"{i}\n")


    # timing the full process of:
    #   - extracting features
    #   - extracting meta features
    console_print("Starting Timing Feature Extraction")
    times_features = []
    src = 'D:\\0 2020 sem 1\\whakaari\\dataset_test_sample.h5'
    dst = 'D:\\0 2020 sem 1\\whakaari\\dataset_test.h5'
    store = 'dataset_test.h5'    
    days = [datetime(2011,1,2), datetime(2012,7,5), datetime(2020,4,10), datetime(2019,12,8)]
    data_list = read_dfs(src, days)
    for i in range(num_times):
        copyfile(src, dst) # create hdf file with data
        start = time()
        source_df = feature_extraction(data_list[0], store, 0.5, 20, 1, n_jobs)
        meta_df = feature_extraction(source_df, store, 0.5, 120, 10, n_jobs, source_win_overlap=0.5, source_obs_per_win=20, source_secs_between_obs=1)
        end = time()
        print()
        os.remove(dst) # delete hdf file created
        times_features.append(end-start)
    
    with open(f'D:\\0 2020 sem 1\\whakaari\\times_features_{num_times}.txt', 'w') as f:
        for i in times_features:
            f.write(f"{i}\n")
    
    with open(f'D:\\0 2020 sem 1\\whakaari\\times_averages_{num_times}.txt', 'w') as f:
        # f.write(f"Full run average: {sum(times_full)/len(times_full)}\n")
        # f.write(f"Download average: {sum(times_dowload)/len(times_dowload)}\n")
        f.write(f"Feature Extration average: {sum(times_features)/len(times_features)}\n")


