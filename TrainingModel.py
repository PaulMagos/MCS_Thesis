import torch
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from typing import Union
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from GRGN.Engines.Generator import Generator
from GRGN.GRGNModel import GRGNModel
from tsl.data import TemporalSplitter
from GRGN.Loss import LogLikelihood, MSE_Custom
from tsl.utils.casting import torch_to_numpy
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader

def get_dataset(dataset_name):
    if dataset_name.lower() == 'airquality':
        return AirQuality(impute_nans=True, small=True)
    elif dataset_name.lower() =='metrla':
        return MetrLA()
    elif dataset_name.lower() =='pemsbay':
        return PemsBay()
    else:
        raise ValueError('Unknown dataset')


class CustomSpatioTemporalDataModule(SpatioTemporalDataModule):
    def train_dataloader(self, shuffle: bool = False,
                         batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('train', shuffle, batch_size)

def run_imputation(model_params, optim, optim_params, dataset_name, dataset_size, model_name, weights_mode, wandb):
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(dataset_name)
    
    # encode time of the day and use it as exogenous variable
    # covariates = {'u': dataset.datetime_encoded('day').values}
    covariates = None

    # get adjacency matrix
    adj = dataset.get_connectivity(**{'method': 'distance',
    'threshold': 0.1,
    'include_self': False,
    'layout': 'edge_index'
    })

    # dataset__ = pd.concat([dataset.dataframe()[-(dataset_size+1):-501], dataset.dataframe()[-dataset_size:-500]], axis=0, ignore_index=True) 
    # dataset__.reset_index(drop=True, inplace=True)
    
    # instantiate dataset
    
    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe()[-dataset_size:],
                                      covariates=covariates,
                                      connectivity=adj,
                                      window=1,
                                      stride=1)
    
    splitter = TemporalSplitter(0.2, 0)

    scalers = {'target': StandardScaler(axis=(0, 1))}
    
    dm = CustomSpatioTemporalDataModule(
        dataset=torch_dataset,
        splitter=splitter,
        scalers=scalers,
        batch_size=1,
        workers=32)
    dm.setup(stage='fit')

    # if cfg.get('in_sample', False):
    dm.trainset = list(range(len(torch_dataset)))
    
    ########################################
    # Generator                            #
    ########################################

    model_cls = GRGNModel

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels)
    

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(model_params)

    loss_fn = LogLikelihood(both=True, weights_mode=weights_mode)

    log_metrics = {
        'Encoder_Loss': LogLikelihood(True, weights_mode=weights_mode),
        'Decoder_Loss': LogLikelihood(False, weights_mode=weights_mode),
        'Mean_Loss': LogLikelihood(both=True, weights_mode=weights_mode),
        'MSE' : MSE_Custom(mixture_weights_mode=weights_mode)
    }

    scheduler_class = getattr(torch.optim.lr_scheduler, 'CosineAnnealingLR')
    scheduler_kwargs = {'eta_min': 0.0001, 'T_max': 50}

    # setup generator
    generator = Generator(model_class=model_cls,
                      model_kwargs=model_kwargs,
                      optim_class=getattr(torch.optim, optim),
                      optim_kwargs=optim_params,
                      loss_fn=loss_fn,
                      metrics=log_metrics,
                      scheduler_class=scheduler_class,
                      scheduler_kwargs=scheduler_kwargs,
                      scale_target=True)

    ########################################
    # logging options                      #
    ########################################
    
    if wandb:
        exp_logger = WandbLogger(project='MCSThesis',
                            name=f'GRGN_{dataset_name}_{dataset_size}_{model_name}')  
    else:
        exp_logger = None

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=10,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs/generation/' + f'{dataset_name}/',
        filename=f'{model_name}' + '-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    trainer = Trainer(
        max_epochs=100,
        default_root_dir='logs/generation/' + f'{dataset_name}/',
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=5,
        callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(generator, datamodule=dm)
    
    output = trainer.predict(generator, dataloaders=dm.val_dataloader(False))
    output = generator.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    _, y_true = (output['y_hat'], output['y'])
    
    y_true = torch.Tensor(y_true)

    input = y_true[-500:-499]
    generation = generator.generate(input, torch.tensor(adj[0]), torch.Tensor(adj[1]), None, 'mean', 5000, both_mean=True, kwargs=dict({'scaler': scalers['target']}))
    generation = generation.reshape(generation.shape[0], generation.shape[-2])
    
    cols = dataset.dataframe().columns
    df = pd.DataFrame(generation, columns=cols)
    df.to_csv(f'./Generations/{dataset_name}/Syntetic{dataset_name}_{dataset_size}_{model_name}_GRGN.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run generation model')
    parser.add_argument('--wandb', action='store_true', 
                            help='Flag to disable Wandb')
    parser.add_argument('--weights_mode', '-w', type=str, choices=['weighted', 'uniform', 'equal_probability'], default='weighted',
                            help='Flag  weights')
    parser.add_argument('--dataset', '-d', type=str, choices=['AirQuality', 'PemsBay', 'MetrLA'], default='AirQuality',
                        help='Name of the Dataset')
    parser.add_argument('--mixture_size', '-m', type=int, default=4,
                        help='Size of the mixtures')
    parser.add_argument('--hidden_size', '-hs', type=int, choices=[4, 8, 16, 32, 64], default=16,
                        help='Size of the Hidden')
    parser.add_argument('--size', '-s', type=int, choices=[1000, 2000, 5000, 8000], default=1000,
                        help='Size of the dataset to use (small or full)')
    parser.add_argument('--learning_rate', '-lr', type=float, choices=[1e-5, 1e-4, 1e-3], default=1e-4,
                        help='Learning rate')
    parser.add_argument('--model_name', '-n', type=str, default='Model',
                        help='Name of the model to use for logging and saving checkpoints')

    args = parser.parse_args()
    
     # Function to check if a value is default
    def check_default(arg_value, default_value):
        return "(default)" if arg_value == default_value else ""

    # Find the length of the longest line
    longest_line = max([
        len(f"  Wandb:            {args.wandb} {check_default(args.wandb, False)}"),
        len(f"  Weights Mode:      {args.weights_mode} {check_default(args.weights_mode, 'weighted')}"),
        len(f"  Dataset:           {args.dataset} {check_default(args.dataset, 'AirQuality')}"),
        len(f"  Mixture Size:      {args.mixture_size} {check_default(args.mixture_size, 4)}"),
        len(f"  Hidden Size:       {args.hidden_size} {check_default(args.hidden_size, 16)}"),
        len(f"  Dataset Size:      {args.size} {check_default(args.size, 1000)}"),
        len(f"  Learning Rate:     {args.learning_rate} {check_default(args.learning_rate, 1e-4)}"),
        len(f"  Model Name:        {args.model_name} {check_default(args.model_name, 'Model')}")
    ])

    # Adjust the line separator based on the longest line
    separator = "=" * longest_line

    # Pretty printing
    print(f"\n{separator}")
    print("Parsed Arguments")
    print(f"{separator}")
    print(f"  Wandb:            {args.wandb} {check_default(args.wandb, False)}")
    print(f"  Weights Mode:      {args.weights_mode} {check_default(args.weights_mode, 'weighted')}")
    print(f"  Dataset:           {args.dataset} {check_default(args.dataset, 'AirQuality')}")
    print(f"  Mixture Size:      {args.mixture_size} {check_default(args.mixture_size, 4)}")
    print(f"  Hidden Size:       {args.hidden_size} {check_default(args.hidden_size, 16)}")
    print(f"  Dataset Size:      {args.size} {check_default(args.size, 1000)}")
    print(f"  Learning Rate:     {args.learning_rate} {check_default(args.learning_rate, 1e-4)}")
    print(f"  Model Name:        {args.model_name} {check_default(args.model_name, 'Model')}")
    print(f"{separator}\n")

    model_params = {
        'hidden_size': args.hidden_size,
        'mixture_size': args.mixture_size,
        'mixture_weights_mode': args.weights_mode
    }
    optim_params = {'lr': args.learning_rate, 'weight_decay': 0.01}
    
    optim = 'RMSprop' # SGD or Adam
    
    res = run_imputation(model_params, optim, optim_params, args.dataset, args.size, args.model_name, args.weights_mode, args.wandb)