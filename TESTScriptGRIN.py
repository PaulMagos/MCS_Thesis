import argparse
import torch
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler, MinMaxScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.engines import Imputer
from tsl.data import TemporalSplitter
import os
import pandas as pd
from GRGN.Loss.LogLikelihood import LogLikelihood
from pytorch_lightning import Trainer
from tsl.utils.casting import torch_to_numpy
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput
from tsl.metrics import torch as torch_metrics
from tsl.metrics import numpy as numpy_metrics
import matplotlib.pyplot as plt
from typing import Optional
from torch.utils.data import DataLoader
import csv

_metrics = {
    'mae': numpy_metrics.mae,
    'mse': numpy_metrics.mse,
    'mape': torch_metrics.mape,
}

class CustomSpatioTemporalDataModule(SpatioTemporalDataModule):
    def train_dataloader(self, shuffle: bool = False,
                         batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('train', False, batch_size)


def save_data(df, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(f'{path}/{name}.csv', index=False)
    
def save_model_data(model_data, path, name):
    with open(f'{path}/{name}.csv', "w", newline="") as f:
        w = csv.DictWriter(f, model_data.keys())
        w.writeheader()
        w.writerow(model_data)
    
def get_dataset(args):
    covariates = None
    p_fault, p_noise = 0., 0.25
    if args.dataset.lower() == 'airquality':
        dataset = AirQuality(impute_nans=True, small=True)
    elif args.dataset.lower() == 'pemsbay':
        if args.mode.lower() == 'i':
            dataset = add_missing_values(PemsBay(),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  seed=56789)
        else:
            dataset = PemsBay()
    elif args.dataset.lower() == 'metrla':
        if args.mode.lower() == 'i':
            dataset = add_missing_values(MetrLA(),
                                    p_fault=p_fault,
                                    p_noise=p_noise,
                                    min_seq=12,
                                    max_seq=12 * 4,
                                    seed=9101112)
        else:
            dataset = MetrLA()
            
    # get adjacency matrix
    adj = dataset.get_connectivity(**{'method': 'distance',
    'threshold': 0.1,
    'include_self': False,
    'layout': 'edge_index'
    })
    
    if args.test_different_adj:
        adj_self_loop = dataset.get_connectivity(**{'method': 'distance',
            'threshold': 0.1,
            'include_self': True,
            'layout': 'edge_index'
            })
        
        adj_sparse = dataset.get_connectivity(**{'method': 'distance',
            'threshold': 0.5,
            'include_self': False,
            'layout': 'edge_index'
            })
    
    if args.mode.lower() == 'i':
        # instantiate dataset
        torch_dataset = ImputationDataset(target=dataset.dataframe()[-args.size:],
                                        mask=dataset.training_mask[-args.size:],
                                        eval_mask=dataset.eval_mask[-args.size:],
                                        covariates=covariates,
                                        transform=MaskInput(),
                                        connectivity=adj,
                                        window=args.seq_length,
                                        stride=1)
    else:
        torch_dataset = SpatioTemporalDataset(target=dataset.dataframe()[-args.size:],
                                        covariates=covariates,
                                        connectivity=adj,
                                        window=args.seq_length,
                                        horizon=1,
                                        stride=1)
    if args.test_different_adj:
        return torch_dataset, dataset.get_splitter(**{'val_len': 0.1, 'test_len': 0.2}), adj, adj_self_loop, adj_sparse, dataset.dataframe().columns.droplevel('channels')
    return torch_dataset, dataset.get_splitter(**{'val_len': 0.1, 'test_len': 0.2}), adj, dataset.dataframe().columns.droplevel('channels')

def load_model(path):
    return Imputer.load_from_checkpoint(path)

def calulate_loss(generator, scaler, data, args, return_true=False):
    loss_fn = LogLikelihood()
    
    output = generator.collate_prediction_outputs(data)
    output = torch_to_numpy(output)
    y_hat, y_true = (output['y_hat'], output['y'])
    mask = torch.tensor(output['mask']) if args.mode.lower() == 'i' else None
    test_true = scaler.transform(torch.tensor(y_true))
    y_hat = torch.tensor(y_hat)
    
    res = dict(loss=loss_fn.bi_loss(y_hat, test_true))
    if return_true:
        return res['loss'], test_true, mask
    return res['loss']
        
def prediction(model, path, model_name, columns, X, Y, edge_index, edge_weight, enc_dec_mean, args, kwargs, adj_name=None):
        model_output = model.predict(X=X, edge_index=edge_index, edge_weight=edge_weight, steps=500, end_dec_mean=enc_dec_mean, **kwargs)
        mae = _metrics['mae'](model_output, Y)
        mse = _metrics['mse'](model_output, Y)
        mape = _metrics['mape'](model_output, Y)
        model_output = pd.DataFrame(model_output[0, :, :, 0])
        model_output.columns = columns
        save_data(model_output, path, model_name + ('_enc_dec_mean' if enc_dec_mean else '') + '_w' + str(args.seq_length) + (f'_{adj_name}' if adj_name is not None else ''))   
        return mae, mse, mape
        
def imputation(model, path, model_name, columns, X, mask, edge_index, edge_weight, enc_dec_mean, args, kwargs, adj_name=None):
        model_output = model.imputation(X=X, mask=mask, edge_index=edge_index, edge_weight=edge_weight, end_dec_mean=enc_dec_mean, **kwargs)
        mae = _metrics['mae'](model_output, X, mask)
        mse = _metrics['mse'](model_output, X, mask)
        mape = _metrics['mape'](model_output, X, mask)
        model_output = pd.DataFrame(model_output[0, :, :, 0])
        model_output.columns = columns
        save_data(model_output, path, model_name + ('_enc_dec_mean' if enc_dec_mean else '') + '_w' + str(args.seq_length) + (f'_{adj_name}' if adj_name is not None else ''))   
        return mae, mse, mape
        
def run(args):
    results_path = 'results/' + args.mode + '/'
    
    if args.test_different_adj:
        torch_dataset, splitter, adj, adj_self, adj_sparse, columns = get_dataset(args)
    else:
        torch_dataset, splitter, adj, columns = get_dataset(args)
    
    scalers = {'target': StandardScaler(axis=(0, 1))}
    dm = CustomSpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=32,
        workers=32)
    dm.setup('test')
    dm.trainset = list(range(len(torch_dataset)))
    
    models_path = args.models_path
    
    trainer = Trainer()
    
    min_los = 0.
    best_model = ''
    models = {el: {'path': os.getcwd() + models_path + el} for el in os.listdir(os.getcwd() + models_path) if el.endswith('.ckpt')}
    for model in models:
        ckpt = load_model(models[model]['path'])      
        for key in ckpt.hparams['model_kwargs']:
            models[model][key] = ckpt.hparams['model_kwargs'][key]
        ckpt.freeze()
        
        output_train = trainer.predict(ckpt, dm.train_dataloader())
        loss = calulate_loss(ckpt, scalers['target'], output_train, args)
        models[model]['train_loss'] = loss
        
        output_val = trainer.predict(ckpt, dm.val_dataloader())
        loss = calulate_loss(ckpt, scalers['target'], output_val, args)
        models[model]['val_loss'] = loss
        
        if min_los > loss:
            min_los = loss
            best_model = model
        
        output_test = trainer.predict(ckpt, dm.test_dataloader())
        loss, y_true, mask = calulate_loss(ckpt, scalers['target'], output_test, args, True)
        models[model]['test_loss'] = loss
        
        kwargs = {'scaler': scalers['target']}
        
        resulting_path = results_path + model.split('ckpt')[0][:-1] + '/'
        modelresulting_name =  model.split('ckpt')[0][:-1]
        edge_index = torch.tensor(adj[0])
        edge_weight = torch.tensor(adj[1])
        if args.mode.lower() == 'i':
            X = y_true[-500:, 0:1].permute(1, 0, 2, 3)
            M = mask[-500:, 0:1].permute(1, 0, 2, 3)
            for enc_dec_mean in [True, False]:
                mae, mse, mape = imputation(ckpt, resulting_path, modelresulting_name, columns, X, M, edge_index, edge_weight, enc_dec_mean, args, kwargs)
                models[model]['imputation_mae' + ('_encdecmean' if enc_dec_mean else '')] = mae
                models[model]['imputation_mse' + ('_encdecmean' if enc_dec_mean else '')] = mse
                models[model]['imputation_mape' + ('_encdecmean' if enc_dec_mean else '')] = mape
            
            if args.test_different_adj:
                for enc_dec_mean in [True, False]:
                    mae, mse, mape = imputation(ckpt, resulting_path, modelresulting_name, columns, X, M, edge_index_self, edge_weight_self, enc_dec_mean, args, kwargs, 'self')
                    models[model]['imputation_mae_self_adj' + ('_encdecmean' if enc_dec_mean else '')] = mae
                    models[model]['imputation_mse_self_adj' + ('_encdecmean' if enc_dec_mean else '')] = mse
                    models[model]['imputation_mape_self_adj' + ('_encdecmean' if enc_dec_mean else '')] = mape
                for enc_dec_mean in [True, False]:
                    mae, mse, mape = imputation(ckpt, resulting_path, modelresulting_name, columns, X, M, edge_index_sparse, edge_weight_sparse, enc_dec_mean, args, kwargs, 'sparse')
                    models[model]['imputation_mae_sparse_adj' + ('_encdecmean' if enc_dec_mean else '')] = mae
                    models[model]['imputation_mse_sparse_adj' + ('_encdecmean' if enc_dec_mean else '')] = mse
                    models[model]['imputation_mape_sparse_adj' + ('_encdecmean' if enc_dec_mean else '')] = mape
            
            save_model_data(models[model], resulting_path, modelresulting_name + '_PARAMS_w' + str(args.seq_length))
        else:
            X = y_true[-999:-899].permute(1, 0, 2, 3)
            Y = y_true[-899:-599].permute(1, 0, 2, 3)
            for enc_dec_mean in [True, False]:
                mae, mse, mape = prediction(ckpt, resulting_path, modelresulting_name, columns, X, Y, edge_index, edge_weight, enc_dec_mean, args, kwargs)
                models[model]['prediction_mae' + ('_encdecmean' if enc_dec_mean else '')] = mae
                models[model]['prediction_mse' + ('_encdecmean' if enc_dec_mean else '')] = mse
                models[model]['prediction_mape' + ('_encdecmean' if enc_dec_mean else '')] = mape
            if args.test_different_adj:
                for enc_dec_mean in [True, False]:
                    mae, mse, mape = prediction(ckpt, resulting_path, modelresulting_name, columns, X, Y, edge_index_self, edge_weight_self, enc_dec_mean, args, kwargs, 'self')                    
                    models[model]['prediction_mae_self_adj' + ('_encdecmean' if enc_dec_mean else '')] = mae
                    models[model]['prediction_mse_self_adj' + ('_encdecmean' if enc_dec_mean else '')] = mse
                    models[model]['prediction_mape_self_adj' + ('_encdecmean' if enc_dec_mean else '')] = mape
                for enc_dec_mean in [True, False]:
                    mae, mse, mape = prediction(ckpt, resulting_path, modelresulting_name, columns, X, Y, edge_index_sparse, edge_weight_sparse, enc_dec_mean, args, kwargs, 'sparse')
                    models[model]['prediction_mae_sparse_adj' + ('_encdecmean' if enc_dec_mean else '')] = mae
                    models[model]['prediction_mse_sparse_adj' + ('_encdecmean' if enc_dec_mean else '')] = mse
                    models[model]['prediction_mape_sparse_adj' + ('_encdecmean' if enc_dec_mean else '')] = mape
            save_model_data(models[model], resulting_path, modelresulting_name + '_PARAMS_w' + str(args.seq_length))
    
    # save a file in the directory named as the best model
    save_model_data(models[best_model], results_path,  'best_PARAMS_w'  + str(args.seq_length))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run generation model')
    parser.add_argument('--nobwd', action='store_true', 
                            help='Flag to disable backward_model')
    parser.add_argument('--dataset', '-d', type=str, choices=['AirQuality', 'PemsBay', 'MetrLA'], default='AirQuality',
                        help='Name of the Dataset')
    parser.add_argument('--seq_length', type=int, default=24,
                        help='Size of the sequence')
    parser.add_argument('--size', '-s', type=int, choices=[5000, 8000], default=5000,
                        help='Size of the dataset to use (small or full)')
    parser.add_argument('--models_path', '-n', type=str, default='/logs/generation/AirQuality/',
                        help='Name of the model to use for logging and saving checkpoints')
    parser.add_argument('-mode', type=str, default='i', choices=['p', 'i'])
    parser.add_argument('--test_different_adj', action='store_true', help='vary the adj matrix')
    
    args = parser.parse_args()
    

    run(args)
    