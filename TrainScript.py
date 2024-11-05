import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from GT import get_dataset
from GT import GTM, GGTM, SGGTM, ASGGTM
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig
from deepecho import PARModel
from omegaconf import DictConfig
from tsl.experiment import Experiment
from tsl.datasets import AirQuality
from tsl.datasets.mts_benchmarks import ExchangeBenchmark
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

MODELS_PATH = f'./models'
GEN_PATH = f'./Datasets/GeneratedDatasets/'
IMAGES_PATH = f'./PNG'
DEVICE = 'cpu'
torch.set_default_device(DEVICE)


def datetime_encoded(index, units) -> pd.DataFrame:
        r"""Transform dataset's temporal index into covariates using sinusoidal
        transformations. Each temporal unit is used as period to compute the
        operations, obtaining two feature (:math:`\sin` and :math:`\cos`) for
        each unit."""
        units = [units]
        index_nano = index.view(np.int64)
        datetime = dict()
        for unit in units:
            nano_unit = pd.Timedelta('1' + unit).value
            nano_sec = index_nano * (2 * np.pi / nano_unit)
            datetime[unit + '_sin'] = np.sin(nano_sec)
            datetime[unit + '_cos'] = np.cos(nano_sec)
        return pd.DataFrame(datetime, index=index, dtype=np.float32)

def get_model_class(model_str):
    if model_str == 'gtm':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(DEVICE)
        model = GTM
    elif model_str == 'ggtm':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(DEVICE)
        model = GGTM
    elif model_str == 'sggtm':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(DEVICE)
        model = SGGTM
    elif model_str == 'asggtm':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(DEVICE)
        model = ASGGTM
    elif model_str == 'dgan':
        model = DGAN
    elif model_str == 'par':
        model = PARModel
    elif model_str == 'gaussianregressor':
        model = GaussianProcessRegressor
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model

def filter_model_args(model_str, model_kwargs, cfg: DictConfig):
    if model_str == 'par':
        return cfg.model.hparams
    elif model_str == 'gaussianregressor':
        if cfg.model.hparams.kernel == "RBF":
            return dict(kernel=RBF(length_scale=cfg.model.hparams.length_scale), random_state=42)
    elif model_str == 'dgan':
        return dict(config=DGANConfig(
            max_sequence_len=cfg.dataset.seq_length,
            sample_len=cfg.dataset.sample_len,
            batch_size=cfg.dataset.batch_size,
            epochs=cfg.model.hparams.epochs,
            apply_feature_scaling=cfg.model.hparams.apply_feature_scaling
        ))
    else:
        model_kwargs.update(cfg.model.hparams)
        return model_kwargs
        

def get_dataset_(dataset_name: str, cfg: DictConfig):
    if dataset_name.startswith('air'):
        max_size = cfg.dataset.max_length
        data = AirQuality(impute_nans=True, small=True)
        if cfg.model.graph:
            adj = data.get_connectivity(**cfg.dataset.connectivity)
            edge_index = torch.tensor(adj[0]).to(torch.int64)
            edge_weight = torch.tensor(adj[1]).to(torch.float)
        # encode time of the day and use it as exogenous variable
        exo = data.datetime_encoded('hour').values[:max_size]
        dataset = data.dataframe()[:max_size]
        train_data = torch.Tensor(get_dataset('AirQuality', dataset, 168)[0])
        
        if cfg.model.name.lower()=='par':
            datetime = dataset.index
            ts_name = np.arange(0, max_size//cfg.dataset.seq_length).reshape(-1, 1).repeat(cfg.dataset.seq_length, 1).reshape(-1)
            dataset.columns = dataset.columns.droplevel('channels')
            dataset.columns = [str(col) for col in dataset.columns]

            cols = list(dataset.columns)
            datetime = dataset.index
            dataset = pd.DataFrame(train_data.reshape(train_data.shape[0]*train_data.shape[1], train_data.shape[2]))
            dataset.columns = cols
            dataset.insert(0, 'TS', ts_name)
            dataset.insert(0, 'Date', datetime)
            return dataset, None, None, None
        
        exo_var = torch.Tensor(exo).reshape(train_data.shape[0], train_data.shape[1], 2)
        
        if cfg.model.graph:
            return train_data.to(DEVICE), exo_var.to(DEVICE), edge_index.to(DEVICE), edge_weight.to(DEVICE)
        return train_data.to(DEVICE), exo_var.to(DEVICE), None, None
    
    if cfg.model.graph:
        raise ValueError(f"Dataset {dataset_name} not available with graph.")
    
    if dataset_name.startswith('exchange'):
        max_size = cfg.dataset.max_length
        data = ExchangeBenchmark()
        # encode time of the day and use it as exogenous variable
        exo = data.datetime_encoded('day').values[:max_size]
        dataset = data.dataframe()[:max_size]
        
        train_data = torch.Tensor(get_dataset('ExchangeBenchmark', dataset, 216)[0])
        if cfg.model.name.lower()=='par':
            datetime = dataset.index
            ts_name = np.arange(0, max_size//cfg.dataset.seq_length).reshape(-1, 1).repeat(cfg.dataset.seq_length, 1).reshape(-1)
            dataset.columns = dataset.columns.droplevel('channels')
            dataset.columns = [str(col) for col in dataset.columns]

            cols = list(dataset.columns)
            datetime = dataset.index
            dataset = pd.DataFrame(train_data.reshape(train_data.shape[0]*train_data.shape[1], train_data.shape[2]))
            dataset.columns = cols
            dataset.insert(0, 'TS', ts_name)
            dataset.insert(0, 'Date', datetime)
            return dataset, None, None, None
        
        exo_var = torch.Tensor(exo).reshape(train_data.shape[0], train_data.shape[1], 2)
        return train_data.to(DEVICE), exo_var.to(DEVICE), None, None
    
    if dataset_name.startswith('synth'):
        train_data = torch.Tensor(get_dataset('Synth', window=63)[0])
        max_size = cfg.dataset.max_length
        if cfg.model.name.lower()=='par':
            ts_name = np.arange(0, max_size//cfg.dataset.seq_length).reshape(-1, 1).repeat(cfg.dataset.seq_length, 1).reshape(-1)

            dataset = pd.DataFrame(train_data.reshape(train_data.shape[0]*train_data.shape[1], train_data.shape[2]))
            dataset.insert(0, 'TS', ts_name)
            return dataset, None, None, None
        
        return train_data.to(DEVICE), None, None, None

    raise ValueError(f"Dataset {dataset_name} not available in this setting.")


def get_new_exo_var(dataset_name: str, max_length: int, num_samples: int):
    if dataset_name.startswith('air'):
        data = AirQuality(impute_nans=True, small=True).dataframe()[:max_length]
        period = 191 * num_samples
        exo_var = pd.date_range(data.index.max() + pd.Timedelta(1, 'Hour'), freq=pd.Timedelta(1, 'Hour'), periods=period)
        exo_var = torch.tensor(datetime_encoded(exo_var, 'hour').values).reshape(num_samples, 191, 2)
        return exo_var.to(DEVICE)
    elif dataset_name.startswith('exchange'):
        data = ExchangeBenchmark().dataframe()[:max_length]
        period = 236 * num_samples
        exo_var = pd.date_range(data.index.max() + pd.Timedelta(1, 'day'), freq='D', periods=period)
        exo_var = torch.tensor(datetime_encoded(exo_var, 'day').values).reshape(num_samples, 236, 2)
        return exo_var.to(DEVICE)
    else:
        return None


def check_path_existance(MODELS_PATH, GEN_PATH, DATASET_NAME, IMAGES_PATH):
    if not os.path.exists(f'{MODELS_PATH}/{DATASET_NAME}/'):
        os.makedirs(f'{MODELS_PATH}/{DATASET_NAME}')
    if not os.path.exists(f'{GEN_PATH}/{DATASET_NAME}/'):
        os.makedirs(f'{GEN_PATH}/{DATASET_NAME}')
    if not os.path.exists(f'{IMAGES_PATH}/{DATASET_NAME}/'):
        os.makedirs(f'{IMAGES_PATH}/{DATASET_NAME}')
        os.makedirs(f'{IMAGES_PATH}/{DATASET_NAME}/History')
        os.makedirs(f'{IMAGES_PATH}/{DATASET_NAME}/Train')
        os.makedirs(f'{IMAGES_PATH}/{DATASET_NAME}/Validation')
        os.makedirs(f'{IMAGES_PATH}/{DATASET_NAME}/Test')

def train(model, dataset, exo_var, cfg: DictConfig):
    if cfg.model.name.lower().endswith('gtm'):
        model, history = model.train_step(dataset, exo_var, batch_size=cfg.dataset.batch_size, window=cfg.dataset.window, horizon=cfg.dataset.horizon, epochs=cfg.model.training.epochs)
        save_model(model, cfg, history)     
    elif cfg.model.name.lower() == 'dgan':
        model.train_numpy(np.array(dataset.cpu()))
        save_model(model, cfg)     
    elif cfg.model.name.lower() == 'par':
        model.fit(data=dataset, segment_size=cfg.dataset.sample_len, entity_columns=['TS'], sequence_index=cfg.dataset.sequence_index)
        save_model(model, cfg)     
    elif cfg.model.name.lower() == 'gaussianregressor':
        train = dataset.reshape(dataset.shape[0]*dataset.shape[1], dataset.shape[2])
        val = train[1:].to('cpu')
        train = train[:-1].to('cpu')
        print('Gaussian Regressor Training Start.')
        model.fit(train, val)
        print('Gaussian Regressor Training End.')
        save_model(model, cfg)     
        
        
def save_model(model, cfg, history=None):
    DATASET_NAME = cfg.dataset.name
    MODEL_NAME = cfg.model.name
    if cfg.model.name.lower().endswith('gtm'):
        torch.save(model.state_dict(), f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}')
        with open(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}.hist', 'w') as hist:
            json.dump(history, hist)
    elif cfg.model.name.lower() == 'dgan':
        model.save(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}')
    elif cfg.model.name.lower() == 'par' or cfg.model.name.lower() == 'gaussianregressor':
        with open(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}','wb') as f:
            pickle.dump(model,f)
            
def load_model(model, cfg):
    DATASET_NAME = cfg.dataset.name
    MODEL_NAME = cfg.model.name
    if cfg.model.name.lower().endswith('gtm'):
        try:
            state_dict = torch.load(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}', weights_only=True)
            model.load_state_dict(state_dict)
        except:
            print('Model not present or incompatible')
            
        with open(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}.hist', 'r') as hist:
            history = json.load(hist)
        return model, history
    elif cfg.model.name.lower() == 'dgan':
        model = DGAN.load(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}')
        return model, None
    elif cfg.model.name.lower() == 'par' or cfg.model.name.lower() == 'gaussianregressor':
        with open(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}', 'rb') as f:
            model = pickle.load(f)
            return model, None
        
def generate(model, dataset, num_samples, cfg):
    if cfg.model.name.lower().endswith('gtm'):
        exo_var = get_new_exo_var(cfg.dataset.name, cfg.dataset.max_length, num_samples)
        input_shape = (num_samples, cfg.dataset.seq_length)
        generated_data = model.generate_step(shape=input_shape, exo_var=exo_var, window=cfg.dataset.window, horizon=cfg.dataset.horizon)
    elif cfg.model.name.lower().endswith('par'):
        generated_data = model.sample(num_entities = num_samples * (cfg.dataset.seq_length//cfg.dataset.sample_len))
        generated_data = torch.Tensor(generated_data.values).reshape(num_samples, cfg.dataset.seq_length, dataset.shape[1])[..., 1:]
    elif cfg.model.name.lower().endswith('dgan'):
        generated_data = torch.Tensor(model.generate_numpy(num_samples)[1])
    elif cfg.model.name.lower().endswith('gaussianregressor'):
        y = dataset.reshape(dataset.shape[0]*dataset.shape[1], dataset.shape[2])[-num_samples*dataset.shape[1]:]
        generated_data = torch.Tensor(model.sample_y(y, n_samples=1).reshape(num_samples, dataset.shape[1], dataset.shape[2]))
    return generated_data
    
def run_imputation(cfg: DictConfig):
    cfg.dataset = cfg.dataset.dataset
    cfg.model = cfg.model.model
    
    DATASET_NAME = cfg.dataset.name
    MODEL_NAME = cfg.model.name
    
    check_path_existance(MODELS_PATH, GEN_PATH, DATASET_NAME, IMAGES_PATH)
    
    ########################################
    # data module                          #
    ########################################
    dataset, exo_var, edge_index, edge_weight = get_dataset_(DATASET_NAME.lower(), cfg)
    
    # encode time of the day and use it as exogenous variable
    input_size = dataset.shape[-1]
    output_size = input_size
    exo_size = exo_var.shape[2] if exo_var is not None else 0

    ########################################
    # Model                                #
    ########################################


    model_kwargs = dict(input_size=input_size,
                        output_size=output_size,
                        exo_size=exo_size,
                        mixture_dim=cfg.dataset.mixture_dim,
                        device=DEVICE)
    
    if cfg.model.graph:
        model_kwargs['edge_index'] = edge_index
        model_kwargs['edge_weight']  = edge_weight


    model_cls = get_model_class(MODEL_NAME.lower())
    
    model_kwargs = filter_model_args(MODEL_NAME.lower(), model_kwargs, cfg)
    
    model = model_cls(**model_kwargs)
    
    ########################################
    # training                             #
    ########################################
    train(model, dataset, exo_var, cfg)

    ########################################
    # testing                              #
    ########################################
    
    model, history = load_model(model, cfg)
    
    generated_data = generate(model, dataset, 100, cfg)
    generated_data = np.array(generated_data).reshape(generated_data.shape[0]*generated_data.shape[1], generated_data.shape[2])
    np.savetxt(f'{GEN_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}.csv', generated_data)
    # return res
    
if __name__ == '__main__':
    exp = Experiment(run_fn=run_imputation, config_path='/data/p.magos/TSGen/config/', config_name='config.yaml')
    res = exp.run()
    # logger.info(res)