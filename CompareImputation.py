# %%
import os
import torch
import numpy as np
import pandas as pd
from GT import get_dataset, denormalize
import matplotlib.pyplot as plt
from tsl.datasets import AirQuality
from tsl.datasets.mts_benchmarks import ExchangeBenchmark
from tsl.metrics.numpy import mase_time, mae, mse, mape

Dataset1 = 'Synthetic'
Dataset2 = 'ExchangeBenchmark'
Dataset3 = 'AirQuality'

masks = {'Synthetic': {}, 'ExchangeBenchmark': {}, 'AirQuality': {}} # Initialize empty dictionaries for each dataset type.
datasets = {'Synthetic': {}, 'ExchangeBenchmark': {}, 'AirQuality': {}} # Initialize empty dictionaries for each dataset type.

dataset3_cols = AirQuality(impute_nans=True, small=True).dataframe().columns.droplevel('channels')
dataset2_cols = ExchangeBenchmark().dataframe().columns.droplevel('channels')
dataset1_cols = ['sin', 'cos', 'sincos', 'scos', 'csin', 'tan']

# %%
# datasets[Dataset1]['Original'] = denormalize(x=get_dataset('Synth', window=63)[0].reshape(-1, 6)).reshape(52, 63, 6)
# datasets[Dataset2]['Original'] = denormalize(x=get_dataset(Dataset2, ExchangeBenchmark().dataframe()[:7560], 216)[0].reshape(-1, 8), name=Dataset2).reshape(35, 216, 8)
# datasets[Dataset3]['Original'] = denormalize(x=get_dataset(Dataset3, AirQuality(impute_nans=True, small=True).dataframe()[:8736], 168)[0].reshape(-1, 36), name=Dataset3).reshape(52, 168, 36)


datasets[Dataset1]['Original'] = get_dataset('Synth', window=63)[0]
datasets[Dataset2]['Original'] = get_dataset(Dataset2, ExchangeBenchmark().dataframe()[:7560], 216)[0]
datasets[Dataset3]['Original'] = get_dataset(Dataset3, AirQuality(impute_nans=True, small=True).dataframe()[:8736], 168)[0]


masks[Dataset1] = {}
masks[Dataset2] = {}
masks[Dataset3] = {}  

# %%
dataset_path = './Datasets/ImputationDatasets/'

for dir in os.listdir(dataset_path):
    if os.path.isdir(f"{dataset_path}/{dir}"):   # If it's a valid folder, iterate through its contents (datasets).
        steps = datasets[dir]['Original'].shape[1]
        for file in os.listdir(f"{dataset_path}/{dir}"):   # Iterate over each file in the directory (each dataset)
            if not file.split('_')[0].endswith('Regressor') and not len(file.split('_'))==1:
                data = np.loadtxt(dataset_path + dir + '/' + file , delimiter = ' ')   # Load the data from each .txt files into a numpy array (each dataset)
                # data = denormalize(data, dir if dir != 'Synthetic' else 'Synth')
                datasets[dir][file.split('_')[0]+'_'+file.split('_')[2].split('.')[0]] = data.reshape(data.shape[0]//steps, steps, data.shape[-1])
            if len(file.split('_'))==1 and 'POINT' not in masks[dir]:
                mask = np.loadtxt(dataset_path + dir + '/' + dir + 'POINTMASK.csv' , delimiter = ' ')
                masks[dir]['POINT'] = mask.reshape(mask.shape[0]//steps, steps, mask.shape[-1])
                if dir != 'AirQuality':
                    mask = np.loadtxt(dataset_path + dir + '/' + dir + 'BLOCKMASK.csv' , delimiter = ' ')
                    masks[dir]['BLOCK'] = mask.reshape(mask.shape[0]//steps, steps, mask.shape[-1])

def calculateVals(datasets: dict, masks: dict, name: str):
    ori = datasets.pop('Original')
    res = []
    for key, data in datasets.items():
        res.append({
                   'dataset': name,
                   'model': key.split('_')[0],
                   'mask': key.split('_')[1], 
                   'mape_score': mape(ori, data, masks[key.split('_')[1]]), 
                   'mae_score': mae(ori, data, masks[key.split('_')[1]]),
                   'mse_score': mse(ori, data, masks[key.split('_')[1]]),
                   'mase_time_score': mase_time(ori, data, masks[key.split('_')[1]])
            }
        )
    return pd.DataFrame(res)


# %%
datasets[Dataset2]['Original'].shape

# %%

plt.plot(datasets[Dataset2]['Original'][1][23:, 2], label='Original')
plt.plot(datasets[Dataset2]['ASGGTM_BLOCK'][1][23:, 2], label='ASGGTM')
# plt.plot(datasets[Dataset2]['GTLSTM_BLOCK'][1][23:, 2], label='LSTM')
# plt.plot(datasets[Dataset3]['GTM'][21][:, 4], label='MDN')
# plt.plot(datasets[Dataset3]['GTR'][1][23:, 6], label='RNN')
plt.legend()

# %%
SYNTH = calculateVals(datasets[Dataset1], masks[Dataset1], Dataset1)
SYNTH
# %%
EXCHANGE = calculateVals(datasets[Dataset2], masks[Dataset2], Dataset2)

# %%
AIRQUALITY = calculateVals(datasets[Dataset3], masks[Dataset3], Dataset3)

data = pd.concat([SYNTH, EXCHANGE, AIRQUALITY], axis = 0)
data.to_csv('imputation_results.csv', index=False)
# %%
