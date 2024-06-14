import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tsl import logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.engines import Imputer
from tsl.experiment import Experiment
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.nn.models import GRINModel
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy

def run_imputation():
    ########################################
    # data module                          #
    ########################################
    dataset = MetrLA()
    
    dataset = add_missing_values(MetrLA(),
                                  p_fault=0.0015,
                                  p_noise=0.05,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  seed=9101112)

    # encode time of the day and use it as exogenous variable
    # covariates = {'u': dataset.datetime_encoded('day').values}
    covariates = None

    # get adjacency matrix
    adj = dataset.get_connectivity(**{'method': 'distance',
    'threshold': 0.1,
    'include_self': False,
    'layout': 'edge_index'
    })

    
    # mymask = torch.full((34272, 207, 1), True, dtype=torch.bool)
    
    shape = (34272, 207, 1)

    # Create a tensor with random values between 0 and 1
    random_tensor = torch.rand(shape)

    # Create a tensor of boolean values where each element is True with a probability of 2%
    mymask = random_tensor < 0.10

    # instantiate dataset
    torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                      mask=mymask,
                                      eval_mask=mymask,
                                      covariates=covariates,
                                      transform=MaskInput(),
                                      connectivity=adj,
                                      window=24,
                                      stride=1)

    scalers = {'target': StandardScaler(axis=(0, 1))}
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(**{'val_len': 0.1, 'test_len': 0.2}),
        batch_size=128,
        workers=128)
    dm.setup(stage='fit')

    # if cfg.get('in_sample', False):
    dm.trainset = list(range(len(torch_dataset)))
    
    ########################################
    # imputer                              #
    ########################################

    model_cls = GRINModel

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,)
                        # exog_size=torch_dataset.input_map.u.shape[-1])
    

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update({'hidden_size': 64,
    'ff_size': 64,
    'embedding_size': 8,
    'n_layers': 1,
    'kernel_size': 2,
    'decoder_order': 1,
    'layer_norm': False,
    'dropout': 0,
    'ff_dropout': 0,
    'merge_mode': 'mlp'})

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mre': torch_metrics.MaskedMRE(),
        'mape': torch_metrics.MaskedMAPE()
    }

    scheduler_class = getattr(torch.optim.lr_scheduler, 'CosineAnnealingLR')
    scheduler_kwargs = {'eta_min': 0.0001, 'T_max': 300}

    # setup imputer
    imputer = Imputer(model_class=model_cls,
                      model_kwargs=model_kwargs,
                      optim_class=getattr(torch.optim, 'Adam'),
                      optim_kwargs={'lr': 0.001, 'weight_decay': 0},
                      loss_fn=loss_fn,
                      metrics=log_metrics,
                      scheduler_class=scheduler_class,
                      scheduler_kwargs=scheduler_kwargs,
                      scale_target=True,
                      whiten_prob=0.05,
                      prediction_loss_weight=1.0,
                      impute_only_missing=False,
                      warm_up_steps=0)

    ########################################
    # logging options                      #
    ########################################
    exp_logger = TensorBoardLogger(save_dir=f'logs/imputation/grin/',
                                       name='tensorboard')

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=50,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs/imputation/grin/',
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(
        max_epochs=300,
        default_root_dir='logs/imputation/grin/',
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=5,
        callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(imputer, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    trainer.test(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    res = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
               test_mre=numpy_metrics.mre(y_hat, y_true, mask),
               test_mape=numpy_metrics.mape(y_hat, y_true, mask))

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    res.update(
        dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
             val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
             val_mape=numpy_metrics.mape(y_hat, y_true, mask)))

    return res


if __name__ == '__main__':
    res = run_imputation()
    # res = exp.run()
    logger.info(res) 