import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from GRGN.Engines.Generator import Generator
from tsl.experiment import Experiment
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from GRGN.GRGNModel import GRGNModel
from GRGN.Loss.LogLikelihood import LogLikelihood
from tsl.ops.imputation import add_missing_values
from tsl.utils.casting import torch_to_numpy

def run_imputation(model_params, optim, optim_params, batch_size):
    ########################################
    # data module                          #
    ########################################
    dataset = PemsBay()
    
    # encode time of the day and use it as exogenous variable
    # covariates = {'u': dataset.datetime_encoded('day').values}
    covariates = None

    # get adjacency matrix
    adj = dataset.get_connectivity(**{'method': 'distance',
    'threshold': 0.1,
    'include_self': False,
    'layout': 'edge_index'
    })

    
    # instantiate dataset
    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                      covariates=covariates,
                                      connectivity=adj,
                                      window=1,
                                      stride=1)

    scalers = {'target': StandardScaler(axis=(0, 1))}
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(**{'val_len': 0.2, 'test_len': 0.1}),
        batch_size=batch_size,
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

    loss_fn = LogLikelihood(both=True)

    log_metrics = {
        '1stLL': LogLikelihood(False),
        '2ndLL': LogLikelihood(),
        '12LL': LogLikelihood(both=True),
    }

    scheduler_class = getattr(torch.optim.lr_scheduler, 'CosineAnnealingLR')
    scheduler_kwargs = {'eta_min': 0.0001, 'T_max': 300}

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
    exp_logger = TensorBoardLogger(save_dir=f'logs/generation/grgn/',
                                       name='tensorboard')

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=50,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs/generation/grgn/',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    trainer = Trainer(
        max_epochs=500,
        default_root_dir='logs/generation/grgn/',
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=5,
        callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(generator, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    generator.load_model(checkpoint_callback.best_model_path)

    generator.freeze()
    trainer.test(generator, datamodule=dm)

    output = trainer.predict(generator, dataloaders=dm.test_dataloader())
    output = generator.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true = (output['y_hat'], output['y'])
    res = dict(test_mae=loss_fn.loss_function(y_hat, y_true))

    output = trainer.predict(generator, dataloaders=dm.val_dataloader())
    output = generator.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true = (output['y_hat'], output['y'])
    res.update(dict(val_mae=loss_fn.loss_function(y_hat, y_true)))
    return res

if __name__ == '__main__':
    model_params = {
        'hidden_size': 32,
        'embedding_size': 16,
        'n_layers': 1,
        'kernel_size': 2,
        'decoder_order': 1,
        'layer_norm': True,
        'dropout': 0.05,
    }
    optim_params = {'lr': 0.00001, 'weight_decay': 0.01}
    
    optim = 'RMSprop' # SGD or Adam
    
    batch_size = 2
    
    res = run_imputation(model_params, optim, optim_params, batch_size)

    logger.info(res)