
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

__all__ = ['get_dataset', 'normalize', 'denormalize']

base_path = f'{os.path.dirname(__file__)}/'

def get_Synth():
    Train = np.load(f'{base_path}/data/Synth/train.npy')
    return Train

def split_data(name, dataset, window):
    if not (os.path.exists(f'{base_path}/data/{name}/train.npy') and os.path.exists(f'{base_path}/data/{name}/preprocessing.npy')):
        Train = dataset
                    
        Train = np.array(Train)
        scaler = MinMaxScaler()
        Train = scaler.fit_transform(Train)
        np.save(f'{base_path}/data/{name}/train.npy', Train)
        np.save(f'{base_path}/data/{name}/scaler.npy', scaler)
    else:
        Train = np.load(f'{base_path}/data/{name}/train.npy')
        
    Train = Train.reshape(Train.shape[0]//window, window, Train.shape[-1])
    return Train
    
    
    
def get_dataset(name='Synth', dataset=None, window=1):
    if name == "Synth":
        train = get_Synth()
    else:
        train = split_data(name, dataset, window=window)
        
    val = train
    test = train
    
    print(f'{name} DATA')
    print(f'Original Dataset: \t{len(test)+len(train)+len(val)}\nTrain Split: \t\t{len(train)} \t(70%)\nValidation Split: \t{len(val)} \t(20%)\nTest Split: \t\t{len(test)} \t(10%)')
    return train, val, test

def normalize(x, name='Synth'):
    scaler = np.load(f'{base_path}/data/{name}/scaler.npy')
    return scaler.transform(x)
    
def denormalize(x, name='Synth'):
    scaler = np.load(f'{base_path}/data/{name}/scaler.npy')
    return scaler.inverse_transform(x)