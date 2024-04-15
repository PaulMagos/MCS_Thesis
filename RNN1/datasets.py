
import os
import pandas as pd
import pickle
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler

__all__ = ['get_dataset']

base_path = f'{os.path.dirname(__file__)}/'
scaler = MinMaxScaler(feature_range=(0,1))

def get_Energy():
    if not os.path.exists(f'{base_path}/data/Energy/train.pkl'):
        data = pd.read_csv(f'{base_path}/data/Energy/LD2011_2014.txt', sep=";")
        data.columns = ['Date'] + list(data.columns[1:])
        data[data.columns[1:]] = data[data.columns[1:]].apply(lambda x: [str(el).replace(',', '.') for el in x]).astype('float64')
        data['Date'] = data['Date'].astype('datetime64[ns]')
        EnergyTrain, EnergyValidation, EnergyTest = \
                    np.split(data,[int(.7*len(data)), int(.9*len(data))])
                    
        pickle.dump(EnergyTrain, open(f'{base_path}/data/Energy/train.pkl', 'wb'))
        pickle.dump(EnergyValidation, open(f'{base_path}/data/Energy/validation.pkl', 'wb'))
        pickle.dump(EnergyTest, open(f'{base_path}/data/Energy/test.pkl', 'wb'))                
    else:
        EnergyTrain = pickle.load(open(f'{base_path}/data/Energy/train.pkl', 'rb'))
        EnergyValidation = pickle.load(open(f'{base_path}/data/Energy/validation.pkl', 'rb'))
        EnergyTest = pickle.load(open(f'{base_path}/data/Energy/test.pkl', 'rb'))
        
    return EnergyTrain, EnergyValidation, EnergyTest
    
def print_upper_line():
    print(f'{''.join(['‾' for _ in range(0, 40)])}')

def print_lower_line():
    print(f'{''.join(['_' for _ in range(0, 40)])}')

def get_EEG():
    if not os.path.exists(f'{base_path}/data/EEG/train.pkl'):
        data = arff.loadarff(f'{base_path}/data/EEG/EEG Eye State.arff')
        data = pd.DataFrame(data[0])
        catCols = [col for col in data.columns if data[col].dtype=="O"]
        data[catCols]=data[catCols].apply(lambda x: x.str.decode('utf8'))
        EEGTrain, EEGValidation, EEGTest = \
                    np.split(data, [int(.7*len(data)), int(.9*len(data))])
                    
        pickle.dump(EEGTrain, open(f'{base_path}/data/EEG/train.pkl', 'wb'))
        pickle.dump(EEGValidation, open(f'{base_path}/data/EEG/validation.pkl', 'wb'))
        pickle.dump(EEGTest, open(f'{base_path}/data/EEG/test.pkl', 'wb'))
    else:
        EEGTrain = pickle.load(open(f'{base_path}/data/EEG/train.pkl', 'rb'))
        EEGValidation = pickle.load(open(f'{base_path}/data/EEG/validation.pkl', 'rb'))
        EEGTest = pickle.load(open(f'{base_path}/data/EEG/test.pkl', 'rb'))
    return EEGTrain, EEGValidation, EEGTest
    
    
def get_dateset(name='EEG'):
    train, val, test = get_EEG() if name == 'EEG' else get_Energy()
    print(f'{name} DATA')
    print_lower_line()
    print(f'Original Dataset: \t{len(test)+len(train)+len(val)}\nTrain Split: \t\t{len(train)} \t(70%)\nValidation Split: \t{len(val)} \t(20%)\nTest Split: \t\t{len(test)} \t(10%)')
    print_upper_line()
    
    train = scaler.fit_transform(np.array(train[train.columns[1:]]))
    val = scaler.fit_transform(np.array(val[val.columns[1:]]))
    test = scaler.fit_transform(np.array(test[test.columns[1:]]))
    return train, val, test