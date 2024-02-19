import pandas as pd
import os

file_path='C:\\Users\\jason.huang\\research\\desciptive_statistics\\pair_instrument\\spread_ratio\\'
data_all = pd.DataFrame()
for file_name in os.listdir(file_path):
    file_location = file_path + file_name
    code = file_name.split('min')[0]
    data = pd.read_csv(file_location)
    data=data.iloc[0,:]
    data=pd.DataFrame(data)
    data=data.T
    data.iloc[0,0]=code
    data_all = data_all.append(data)

data_all.to_csv("statinary_ratio.csv")