
import os
import pandas as pd
import pickle
import numpy as np
from scipy.io import arff

__all__ = ['get_dataset', 'normalize', 'denormalize']

base_path = f'{os.path.dirname(__file__)}/'

def get_SynteticSin(stds_to_use: int, change: bool, version='SynteticSin'):
    if not (os.path.exists(f'{base_path}/data/{version}/train.pkl') and os.path.exists(f'{base_path}/data/{version}/preprocessing.npz')):
        data = pd.read_json(f'{base_path}/data/{version}/data.json')
        SinTrain, SinValidation, SinTest = \
                    np.split(data,[int(.7*len(data)), int(.9*len(data))])
        pickle.dump(SinTrain, open(f'{base_path}/data/{version}/train.pkl', 'wb'))
        pickle.dump(SinValidation, open(f'{base_path}/data/{version}/validation.pkl', 'wb'))
        pickle.dump(SinTest, open(f'{base_path}/data/{version}/test.pkl', 'wb'))            
        means = np.mean(np.array(data[data.columns]), 0)
        stds = stds_to_use * np.std(np.array(data[data.columns]), 0)
        np.savez(f'{base_path}/data/{version}/preprocessing.npz', means=means, stds=stds, change=change)
    else:
        SinTrain = pickle.load(open(f'{base_path}/data/{version}/train.pkl', 'rb'))
        SinValidation = pickle.load(open(f'{base_path}/data/{version}/validation.pkl', 'rb'))
        SinTest = pickle.load(open(f'{base_path}/data/{version}/test.pkl', 'rb'))
    
    SinTrain = normalize(name=version, x=np.array(SinTrain[SinTrain.columns]))
    SinValidation = normalize(name=version, x=np.array(SinValidation[SinValidation.columns]))
    SinTest = normalize(name=version, x=np.array(SinTest[SinTest.columns]))
    SinTrain = denormalize(name=version, x=SinTrain)
    SinValidation = denormalize(name=version, x=SinValidation)
    SinTest = denormalize(name=version, x=SinTest)
    
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

def split_data(name, dataset, stds_to_use):
    if not (os.path.exists(f'{base_path}/data/{name}/train.pkl') and os.path.exists(f'{base_path}/data/{name}/preprocessing.npz')):
        Train, Validation, Test = \
                    np.split(dataset, [int(.9*len(dataset)), int(.95*len(dataset))])
                    
        Train = np.array(Train)
        Validation = np.array(Validation)
        Test = np.array(Test)
                    
        pickle.dump(Train, open(f'{base_path}/data/{name}/train.pkl', 'wb'))
        pickle.dump(Validation, open(f'{base_path}/data/{name}/validation.pkl', 'wb'))
        pickle.dump(Test, open(f'{base_path}/data/{name}/test.pkl', 'wb'))
        means = np.mean(dataset, 0)
        stds = stds_to_use * np.std(dataset, 0)
        np.savez(f'{base_path}/data/{name}/preprocessing.npz', means=means, stds=stds, change=True)    
    else:
        Train = pickle.load(open(f'{base_path}/data/{name}/train.pkl', 'rb'))
        Validation = pickle.load(open(f'{base_path}/data/{name}/validation.pkl', 'rb'))
        Test = pickle.load(open(f'{base_path}/data/{name}/test.pkl', 'rb'))
        
        
    Train = normalize(name=name, x=np.array(Train))
    Validation = normalize(name=name, x=np.array(Validation))
    Test = normalize(name=name, x=np.array(Test))
    
    Train = Train.reshape(Train.shape[0], 1, Train.shape[-1])
    Validation = Validation.reshape(Validation.shape[0], 1, Validation.shape[-1])
    Test = Test.reshape(Test.shape[0], 1, Test.shape[-1])
    return Train, Validation, Test
    
    
    
def get_dataset(name='EEG', dataset=None, stds_to_use=10, change=True):
    match(name):
        case 'EEG':
            train, val, test = get_EEG(stds_to_use, change)
        case 'Energy':
            train, val, test = get_Energy(stds_to_use, change)
        case 'SynteticSin':
            train, val, test = get_SynteticSin(stds_to_use, change)
        case 'SynteticSin2':
            train, val, test = get_SynteticSin(stds_to_use, change, version='SynteticSin2')
        case _:
            train, val, test = split_data(name, dataset, stds_to_use)
    
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