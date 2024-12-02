# %%
import os
import torch
import numpy as np
import pandas as pd
from GT import get_dataset
import matplotlib.pyplot as plt
from tsl.datasets import AirQuality
from tsl.datasets.mts_benchmarks import ExchangeBenchmark
from tsl.metrics.numpy import mae, mase_time, mape, mse

Dataset1 = 'Synthetic'
Dataset2 = 'ExchangeBenchmark'
Dataset3 = 'AirQuality'

datasets = {'Synthetic': {}, 'ExchangeBenchmark': {}, 'AirQuality': {}} # Initialize empty dictionaries for each dataset type.

dataset3_cols = AirQuality(impute_nans=True, small=True).dataframe().columns.droplevel('channels')
dataset2_cols = ExchangeBenchmark().dataframe().columns.droplevel('channels')
dataset1_cols = ['sin', 'cos', 'sincos', 'scos', 'csin', 'tan']

# %%
datasets[Dataset1]['Original'] = get_dataset('Synth', window=63)[0]
datasets[Dataset2]['Original'] = get_dataset(Dataset2, ExchangeBenchmark().dataframe()[:7560], 216)[0]
datasets[Dataset3]['Original'] = get_dataset(Dataset3, AirQuality(impute_nans=True, small=True).dataframe()[:8736], 168)[0]

# %%
dataset_path = './Datasets/PredictionDatasets/'

Order = ['Original', 'GTR', 'GTLSTM', 'GTM', 'GGTM', 'SGGTM', 'ASGGTM', 'ASGGTM2']

for dir in os.listdir(dataset_path):
    if os.path.isdir(f"{dataset_path}/{dir}"):   # If it's a valid folder, iterate through its contents (datasets).
        steps = datasets[dir]['Original'].shape[1]
        for file in os.listdir(f"{dataset_path}/{dir}"):   # Iterate over each file in the directory (each dataset)
            if not file.split('_')[0].endswith('Regressor'):
                data = np.loadtxt(dataset_path + dir + '/' + file , delimiter = ' ')   # Load the data from each .txt files into a numpy array (each dataset)
                datasets[dir][file.split('_')[0]] = data.reshape(data.shape[0]//steps, steps, data.shape[-1])
                # datasets[dir][file.split('_')[0]] = data.reshape(data.shape[0]//steps, steps, data.shape[-1])
                
        datasets[dir] = {key: datasets[dir][key] for key in Order if key in datasets[dir]}
        

def calculateVals(datasets: dict, name: str):
    ori = datasets.pop('Original')
    res = []
    for key, data in datasets.items():
        res.append({
                   'dataset': name,
                   'model': key,
                   'mape_score': np.round(mape(ori, data), 4), 
                   'mae_score': np.round(mae(ori, data), 4),
                   'mse_score': np.round(mse(ori, data), 4),
                   'mase_time_score': np.round(mase_time(ori, data), 4)
            }
        )
    return pd.DataFrame(res)


# %%
datasets[Dataset2]['Original'].shape

# %%

plt.plot(datasets[Dataset2]['Original'][1][20:, 3], label='Original')
plt.plot(datasets[Dataset2]['ASGGTM'][1][20:, 3], label='ASGTM')
# plt.plot(datasets[Dataset1]['GTLSTM'][1][15:, 5], label='LSTM')
# plt.plot(datasets[Dataset1]['GTM'][1][15:, 5], label='MDN')
plt.plot(datasets[Dataset2]['GTR'][1][20:, 3], label='RNN')
plt.legend()

# %%
SYNTH = calculateVals(datasets[Dataset1], Dataset1)

# %%
EXCHANGE = calculateVals(datasets[Dataset2], Dataset2)

# %%
AIRQUALITY = calculateVals(datasets[Dataset3], Dataset3)

data = pd.concat([SYNTH, EXCHANGE, AIRQUALITY], axis = 0)
data.to_csv('prediction_results.csv', index=False)
# %%
