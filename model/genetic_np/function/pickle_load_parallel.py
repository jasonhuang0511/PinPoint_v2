import pickle
from multiprocessing import Pool


def read_single_pickle(file_location):
    with open(file_location, 'rb') as file:
        data = pickle.load(file)
    return data


def pickle_load_parallel(file_location_list):
    with Pool(processes=10) as pool:
        # have your pool map the file names to dataframes
        df_list = pool.map(read_single_pickle, file_location_list)
    return df_list
