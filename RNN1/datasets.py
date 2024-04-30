
import os
import pandas as pd
import pickle
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler

__all__ = ['get_dataset', 'inverse_transform']

base_path = f'{os.path.dirname(__file__)}/'

energy_scaler = MinMaxScaler(feature_range=(0,1))
eeg_scaler = MinMaxScaler(feature_range=(0,1))

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
        
    EnergyTrain = energy_scaler.fit_transform(np.array(EnergyTrain[EnergyTrain.columns[1:]]))
    EnergyValidation = energy_scaler.fit_transform(np.array(EnergyValidation[EnergyValidation.columns[1:]]))
    EnergyTest = energy_scaler.fit_transform(np.array(EnergyTest[EnergyTest.columns[1:]]))
    return EnergyTrain, EnergyValidation, EnergyTest
    
# def print_upper_line():
#     print(f'{''.join(['_' for _ in range(0, 40)])}')

def print_lower_line():
    print(f'{"".join(["_" for _ in range(0, 40)])}')

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
    
    EEGTrain = eeg_scaler.fit_transform(np.array(EEGTrain[EEGTrain.columns[1:]]))
    EEGValidation = eeg_scaler.fit_transform(np.array(EEGValidation[EEGValidation.columns[1:]]))
    EEGTest = eeg_scaler.fit_transform(np.array(EEGTest[EEGTest.columns[1:]]))
    
    return EEGTrain, EEGValidation, EEGTest
    
    
def get_dateset(name='EEG'):
    train, val, test = get_EEG() if name == 'EEG' else get_Energy()
    print(f'{name} DATA')
    print_lower_line()
    print(f'Original Dataset: \t{len(test)+len(train)+len(val)}\nTrain Split: \t\t{len(train)} \t(70%)\nValidation Split: \t{len(val)} \t(20%)\nTest Split: \t\t{len(test)} \t(10%)')
    print_lower_line()
    return train, val, test

def inverse_transform(data, name='EEG'):
    if name == 'EEG':
        return eeg_scaler.inverse_transform(data)
    elif name == 'Energy':
        return energy_scaler.inverse_transform(data)
