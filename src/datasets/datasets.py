
import os
import pandas as pd
import pickle
import numpy as np
from scipy.io import arff

__all__ = ['get_dataset', 'normalize', 'denormalize']

base_path = f'{os.path.dirname(__file__)}/'

def get_SynteticSin(stds_to_use: int, change: bool):
    if not (os.path.exists(f'{base_path}/data/SynteticSin/train.pkl') and os.path.exists(f'{base_path}/data/SynteticSin/preprocessing.npz')):
        data = pd.read_json(f'{base_path}/data/SynteticSin/data.json')
        SinTrain, SinValidation, SinTest = \
                    np.split(data,[int(.7*len(data)), int(.9*len(data))])
        pickle.dump(SinTrain, open(f'{base_path}/data/SynteticSin/train.pkl', 'wb'))
        pickle.dump(SinValidation, open(f'{base_path}/data/SynteticSin/validation.pkl', 'wb'))
        pickle.dump(SinTest, open(f'{base_path}/data/SynteticSin/test.pkl', 'wb'))            
        means = np.mean(np.array(data[data.columns]), 0)
        stds = stds_to_use * np.std(np.array(data[data.columns]), 0)
        np.savez(f'{base_path}/data/SynteticSin/preprocessing.npz', means=means, stds=stds, change=change)
    else:
        SinTrain = pickle.load(open(f'{base_path}/data/SynteticSin/train.pkl', 'rb'))
        SinValidation = pickle.load(open(f'{base_path}/data/SynteticSin/validation.pkl', 'rb'))
        SinTest = pickle.load(open(f'{base_path}/data/SynteticSin/test.pkl', 'rb'))
    
    SinTrain = normalize(name='SynteticSin', x=np.array(SinTrain[SinTrain.columns]))
    SinValidation = normalize(name='SynteticSin', x=np.array(SinValidation[SinValidation.columns]))
    SinTest = normalize(name='SynteticSin', x=np.array(SinTest[SinTest.columns]))
    SinTrain = denormalize(name='SynteticSin', x=SinTrain)
    SinValidation = denormalize(name='SynteticSin', x=SinValidation)
    SinTest = denormalize(name='SynteticSin', x=SinTest)
    
    SinTrain = SinTrain.reshape(SinTrain.shape[0], 1, SinTrain.shape[-1])
    SinValidation = SinValidation.reshape(SinValidation.shape[0], 1, SinValidation.shape[-1])
    SinTest = SinTest.reshape(SinTest.shape[0], 1, SinTest.shape[-1])
    return SinTrain, SinValidation, SinTest

def get_Energy(stds_to_use: int, change: bool):
    if not (os.path.exists(f'{base_path}/data/Energy/train.pkl') and os.path.exists(f'{base_path}/data/Energy/preprocessing.npz')):
        data = pd.read_csv(f'{base_path}/data/Energy/LD2011_2014.txt', sep=";")
        data.columns = ['Date'] + list(data.columns[1:])
        data[data.columns[1:]] = data[data.columns[1:]].apply(lambda x: [str(el).replace(',', '.') for el in x]).astype('float64')
        data['Date'] = data['Date'].astype('datetime64[ns]')
        EnergyTrain, EnergyValidation, EnergyTest = \
                    np.split(data,[int(.7*len(data)), int(.9*len(data))])
                    
        pickle.dump(EnergyTrain, open(f'{base_path}/data/Energy/train.pkl', 'wb'))
        pickle.dump(EnergyValidation, open(f'{base_path}/data/Energy/validation.pkl', 'wb'))
        pickle.dump(EnergyTest, open(f'{base_path}/data/Energy/test.pkl', 'wb'))            
        means = np.mean(np.array(data[data.columns[1:]]), 0)
        stds = stds_to_use * np.std(np.array(data[data.columns[1:]]), 0)
        np.savez(f'{base_path}/data/Energy/preprocessing.npz', means=means, stds=stds, change=change)    
    else:
        EnergyTrain = pickle.load(open(f'{base_path}/data/Energy/train.pkl', 'rb'))
        EnergyValidation = pickle.load(open(f'{base_path}/data/Energy/validation.pkl', 'rb'))
        EnergyTest = pickle.load(open(f'{base_path}/data/Energy/test.pkl', 'rb'))
    
        
    EnergyTrain = normalize(name='Energy', x=np.array(EnergyTrain[EnergyTrain.columns[1:]]))
    EnergyValidation = normalize(name='Energy', x=np.array(EnergyValidation[EnergyValidation.columns[1:]]))
    EnergyTest = normalize(name='Energy', x=np.array(EnergyTest[EnergyTest.columns[1:]]))
    EnergyTrain = denormalize(name='Energy', x=EnergyTrain)
    EnergyValidation = denormalize(name='Energy', x=EnergyValidation)
    EnergyTest = denormalize(name='Energy', x=EnergyTest)
    
    EnergyTrain = EnergyTrain.reshape(EnergyTrain.shape[0], 1, EnergyTrain.shape[-1])
    EnergyValidation = EnergyValidation.reshape(EnergyValidation.shape[0], 1, EnergyValidation.shape[-1])
    EnergyTest = EnergyTest.reshape(EnergyTest.shape[0], 1, EnergyTest.shape[-1])
    return EnergyTrain, EnergyValidation, EnergyTest

def print_line():
    print(f'{"".join(["_" for _ in range(0, 40)])}')

def get_EEG(stds_to_use: int, change: bool):
    if not (os.path.exists(f'{base_path}/data/EEG/train.pkl') and os.path.exists(f'{base_path}/data/EEG/preprocessing.npz')):
        data = arff.loadarff(f'{base_path}/data/EEG/EEG Eye State.arff')
        data = pd.DataFrame(data[0])
        catCols = [col for col in data.columns if data[col].dtype=="O"]
        data[catCols]=data[catCols].apply(lambda x: x.str.decode('utf8'))
        EEGTrain, EEGValidation, EEGTest = \
                    np.split(data, [int(.7*len(data)), int(.9*len(data))])
                    
        pickle.dump(EEGTrain, open(f'{base_path}/data/EEG/train.pkl', 'wb'))
        pickle.dump(EEGValidation, open(f'{base_path}/data/EEG/validation.pkl', 'wb'))
        pickle.dump(EEGTest, open(f'{base_path}/data/EEG/test.pkl', 'wb'))
        data = np.array(data[data.columns[:-1]])
        means = np.mean(data, 0)
        stds = stds_to_use * np.std(data, 0)
        np.savez(f'{base_path}/data/EEG/preprocessing.npz', means=means, stds=stds, change=change)    
    else:
        EEGTrain = pickle.load(open(f'{base_path}/data/EEG/train.pkl', 'rb'))
        EEGValidation = pickle.load(open(f'{base_path}/data/EEG/validation.pkl', 'rb'))
        EEGTest = pickle.load(open(f'{base_path}/data/EEG/test.pkl', 'rb'))
    
    EEGTrain = normalize(name='EEG', x=np.array(EEGTrain[EEGTrain.columns[:-1]]))
    EEGValidation = normalize(name='EEG', x=np.array(EEGValidation[EEGValidation.columns[:-1]]))
    EEGTest = normalize(name='EEG', x=np.array(EEGTest[EEGTest.columns[:-1]]))
    EEGTrain = denormalize(name='EEG', x=EEGTrain)
    EEGValidation = denormalize(name='EEG', x=EEGValidation)
    EEGTest = denormalize(name='EEG', x=EEGTest)
    
    EEGTrain = EEGTrain.reshape(EEGTrain.shape[0], 1, EEGTrain.shape[-1])
    EEGValidation = EEGValidation.reshape(EEGValidation.shape[0], 1, EEGValidation.shape[-1])
    EEGTest = EEGTest.reshape(EEGTest.shape[0], 1, EEGTest.shape[-1])
    return EEGTrain, EEGValidation, EEGTest
    
    
def get_dateset(name='EEG', stds_to_use=10, change=True):
    match(name):
        case 'EEG':
            train, val, test = get_EEG(stds_to_use, change)
        case 'Energy':
            train, val, test = get_Energy(stds_to_use, change)
        case 'SynteticSin':
            train, val, test = get_SynteticSin(stds_to_use, change)
    
    print(f'{name} DATA')
    print_line()
    print(f'Original Dataset: \t{len(test)+len(train)+len(val)}\nTrain Split: \t\t{len(train)} \t(70%)\nValidation Split: \t{len(val)} \t(20%)\nTest Split: \t\t{len(test)} \t(10%)')
    print_line()
    return train, val, test

def normalize(x, name='EEG'):
    preprocessing = np.load(f'{base_path}/data/{name}/preprocessing.npz')
    means, stds, change =  preprocessing['means'], preprocessing['stds'], preprocessing['change']
    return np.nan_to_num((x - means) / stds)
    
def denormalize(x, name='EEG'):
    preprocessing = np.load(f'{base_path}/data/{name}/preprocessing.npz')
    means, stds, change =  preprocessing['means'], preprocessing['stds'], preprocessing['change']
    return x * stds + means