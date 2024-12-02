import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from GT import get_dataset
from GT import GTM, GTLSTM, GTR, GGTM, SGGTM, ASGGTM
import torch.nn.functional as F
from omegaconf import DictConfig
from tsl.experiment import Experiment
from tsl.datasets import AirQuality
from tsl.datasets.mts_benchmarks import ExchangeBenchmark
from tsl.ops.imputation import add_missing_values, sample_mask

MODELS_PATH = f'./models'
GEN_PATH = f'./Datasets/ImputationDatasets/'
IMAGES_PATH = f'./PNG'
DEVICE = 'cpu'

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
        model = GTM   
    elif model_str == 'ggtm':
        model = GGTM
    elif model_str == 'sggtm':
        model = SGGTM
    elif model_str == 'asggtm':
        model = ASGGTM
    elif model_str == 'gtlstm':
        model = GTLSTM
    elif model_str == 'gtr':
        model = GTR
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model

def filter_model_args(model_kwargs, cfg: DictConfig):
    if cfg.model.name.lower() == 'gtlstm' or cfg.model.name.lower() == 'gtr':
        model_kwargs.pop('mixture_dim')
        
    model_kwargs.update(cfg.model.hparams)
    return model_kwargs

def get_dataset_(dataset_name: str, cfg: DictConfig):
    if dataset_name.startswith('air'):
        max_size = cfg.dataset.max_length
        data = AirQuality(impute_nans=True, small=True)
        mask = data.mask
        if cfg.model.graph:
            adj = data.get_connectivity(**cfg.dataset.connectivity)
            edge_index = torch.tensor(adj[0]).to(torch.int64)
            edge_weight = torch.tensor(adj[1]).to(torch.float)
        # encode time of the day and use it as exogenous variable
        exo = data.datetime_encoded('hour').values[:max_size]
        mask = mask[:max_size]
        mask = (~torch.Tensor(mask).bool()).float().numpy()
        np.savetxt(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}POINTMASK.csv', mask.squeeze(-1))
        dataset = data.dataframe()[:max_size]
        train_data = torch.Tensor(get_dataset('AirQuality', dataset, 168)[0])
        mask = torch.Tensor(mask).reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])
        exo_var = torch.Tensor(exo).reshape(train_data.shape[0], train_data.shape[1], 2)
        if cfg.model.graph:
            return train_data.to(DEVICE), exo_var.to(DEVICE), edge_index.to(DEVICE), edge_weight.to(DEVICE), mask, None
       
        return train_data.to(DEVICE), exo_var.to(DEVICE), None, None, mask, None
    
    if cfg.model.graph:
        raise ValueError(f"Dataset {dataset_name} not available with graph.")
    
    if dataset_name.startswith('exchange'):
        max_size = cfg.dataset.max_length
        data = ExchangeBenchmark()
        exo = data.datetime_encoded('day').values[:max_size]
        dataset = data.dataframe()[:max_size]
        train_data = torch.Tensor(get_dataset('ExchangeBenchmark', dataset, 216)[0])
        
        
        if not os.path.exists(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}POINTMASK.csv'):
            mask_point = sample_mask(shape=(train_data.shape[0]*train_data.shape[1], train_data.shape[2], 1),
                                     p_noise=0.25, p=0., min_seq=12, max_seq=12*4, rng=np.random.default_rng(42))
            mask_block = sample_mask(shape=(train_data.shape[0]*train_data.shape[1], train_data.shape[2], 1),
                                     p_noise=0.05, p=0.0015, min_seq=12, max_seq=12*4, rng=np.random.default_rng(42))
            mask_point = mask_point[:max_size].squeeze(-1)
            mask_block = mask_block[:max_size].squeeze(-1)
            print(mask_point.shape)
            np.savetxt(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}POINTMASK.csv', mask_point)
            np.savetxt(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}BLOCKMASK.csv', mask_block)
        else:
            mask_point = np.loadtxt(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}POINTMASK.csv')
            mask_block = np.loadtxt(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}BLOCKMASK.csv')
        # encode time of the day and use it as exogenous variable
        
        mask_point = torch.Tensor(mask_point).reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])
        mask_block = torch.Tensor(mask_block).reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])
        exo_var = torch.Tensor(exo).reshape(train_data.shape[0], train_data.shape[1], 2)
        return train_data.to(DEVICE), exo_var.to(DEVICE), None, None, mask_point, mask_block   
    
    if dataset_name.startswith('synth'):
        train_data = torch.Tensor(get_dataset('Synth', window=63)[0])
        
        if not os.path.exists(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}POINTMASK.csv'):
            mask_point = sample_mask(shape=(train_data.shape[0]*train_data.shape[1], train_data.shape[2], 1),
                                    p=0.,
                                    p_noise=0.25,
                                    min_seq=12,
                                    max_seq=12*4,
                                    rng=np.random.default_rng(42)
                                    )
            mask_block = sample_mask(shape=(train_data.shape[0]*train_data.shape[1], train_data.shape[2], 1),
                                    p=0.0015,
                                    p_noise=0.5,
                                    min_seq=12,
                                    max_seq=12*4,
                                    rng=np.random.default_rng(42)
                                    )
            max_size = cfg.dataset.max_length
            mask_point = mask_point[:max_size].squeeze(-1)
            mask_block = mask_block[:max_size].squeeze(-1)
            np.savetxt(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}POINTMASK.csv', mask_point)
            np.savetxt(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}BLOCKMASK.csv', mask_block)
        else:
            mask_point = np.loadtxt(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}POINTMASK.csv')
            mask_block = np.loadtxt(f'{GEN_PATH}/{cfg.dataset.name}/{cfg.dataset.name}BLOCKMASK.csv')
            
        mask_point = torch.Tensor(mask_point).reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])
        mask_block = torch.Tensor(mask_block).reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])
        return train_data.to(DEVICE), None, None, None, mask_point, mask_block

    raise ValueError(f"Dataset {dataset_name} not available in this setting.")


def get_new_exo_var(dataset_name: str, max_length: int, num_samples: int):
    if dataset_name.lower().startswith('air'):
        data = AirQuality(impute_nans=True, small=True).dataframe()[:max_length]
        period = 191 * num_samples
        exo_var = pd.date_range(data.index.max() + pd.Timedelta(1, 'Hour'), freq=pd.Timedelta(1, 'Hour'), periods=period)
        exo_var = torch.tensor(datetime_encoded(exo_var, 'hour').values).reshape(num_samples, 191, 2)
        return exo_var.to(DEVICE)
    elif dataset_name.lower().startswith('exchange'):
        data = ExchangeBenchmark().dataframe()[:max_length]
        period = 236 * num_samples
        exo_var = pd.date_range(data.index.min() + pd.Timedelta(1, 'day'), freq='D', periods=period)
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
    model, history = model.train_step(dataset, exo_var, batch_size=cfg.dataset.batch_size, epochs=cfg.model.training.epochs)
    save_model(model, cfg, history)     
        
def save_model(model, cfg, history=None):
    DATASET_NAME = cfg.dataset.name
    MODEL_NAME = cfg.model.name
    torch.save(model.state_dict(), f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}')
    with open(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}.hist', 'w') as hist:
        json.dump(history, hist)
        
def load_model(model, cfg):
    DATASET_NAME = cfg.dataset.name
    MODEL_NAME = cfg.model.name
    try:
        state_dict = torch.load(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}', weights_only=True)
        model.load_state_dict(state_dict)
    except:
        print('Model not present or incompatible')
        
    with open(f'{MODELS_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}.hist', 'r') as hist:
        history = json.load(hist)
    return model, history
        
def impute(model, dataset, mask, exo_var, cfg):
    generated_data = model.predict_step(dataset, mask, exo_var)
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
    dataset, exo_var, edge_index, edge_weight, mask_point, mask_block = get_dataset_(DATASET_NAME.lower(), cfg)
    
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
                        device=cfg.model.device,
                        window=cfg.dataset.window, 
                        horizon=cfg.dataset.horizon)
    
    if cfg.model.graph:
        model_kwargs['edge_index'] = edge_index      
        model_kwargs['edge_weight']  = edge_weight

    if cfg.model.embedding:
        model_kwargs['emb_size'] = cfg.dataset.emb_size


    model_cls = get_model_class(MODEL_NAME.lower())
    
    
    model_kwargs = filter_model_args(model_kwargs, cfg)
    
    model = model_cls(**model_kwargs)
    
    ########################################
    # training                             #
    ########################################
    # train(model, dataset, exo_var, cfg)

    ########################################
    # testing                              #
    ########################################
    
    model, history = load_model(model, cfg)
    
    # if cfg.model.name=='ASGGTM':
    #     E1 = model.N1.cpu().detach()
    #     E2 = model.N2.cpu().detach()
    #     adp = F.softmax(F.relu(torch.mm(E1, E2)), dim=1).cpu().detach()
    #     adp = np.array(adp)
    #     np.savetxt(f'{GEN_PATH}/{MODEL_NAME}_{DATASET_NAME}_ADJ.csv', adp)   
    generated_data_point = impute(model, dataset, mask_point, exo_var, cfg)
    generated_data_point = np.array(generated_data_point).reshape(generated_data_point.shape[0]*generated_data_point.shape[1], generated_data_point.shape[2])
    np.savetxt(f'{GEN_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}_POINT.csv', generated_data_point)
    if cfg.dataset.name.lower()!= 'airquality':
        generated_data_block = impute(model, dataset, mask_block, exo_var, cfg)
        generated_data_block = np.array(generated_data_block).reshape(generated_data_block.shape[0]*generated_data_block.shape[1], generated_data_block.shape[2])
        np.savetxt(f'{GEN_PATH}/{DATASET_NAME}/{MODEL_NAME}_{DATASET_NAME}_BLOCK.csv', generated_data_block)
    # return res
    
if __name__ == '__main__':
    exp = Experiment(run_fn=run_imputation, config_path='/data/p.magos/TSGen/config/', config_name='config.yaml')
    res = exp.run()
    # logger.info(res)