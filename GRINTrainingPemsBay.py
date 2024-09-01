import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tsl import logger
from tsl.nn.models import GRINModel
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput
from tsl.engines import Imputer
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.utils.casting import torch_to_numpy

def run_imputation(model_params, optim, optim_params, batch_size):
    
    p_fault, p_noise = 0.0015, 0.05
    dataset = add_missing_values(PemsBay(),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  seed=56789)
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
    torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                      mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      covariates=covariates,
                                      transform=MaskInput(),
                                      connectivity=adj,
                                      window=1,
                                      stride=1)

    scalers = {'target': StandardScaler(axis=(0, 1))}
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(**{'val_len': 0.2, 'test_len': 0.1}),
        batch_size=batch_size,
        workers=8)
    dm.setup(stage='fit')
    dm.trainset = list(range(len(torch_dataset)))

    ########################################
    # predictor                            #
    ########################################

    model_cls = GRINModel

    model_kwargs = dict(
                            n_nodes=torch_dataset.n_nodes,
                            input_size=torch_dataset.n_channels,
                            # exog_size=torch_dataset.input_map.u.shape[-1]
                        )

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(model_params)

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mape': torch_metrics.MaskedMAPE(),
    }
    scheduler_class = getattr(torch.optim.lr_scheduler, 'CosineAnnealingLR')
    scheduler_kwargs = {'eta_min': 0.0001, 'T_max': 300}
    
    imputer = Imputer(model_class=model_cls,
                    model_kwargs=model_kwargs,
                    optim_class=getattr(torch.optim, optim),
                    optim_kwargs=dict(optim_params),
                    loss_fn=loss_fn,
                    metrics=log_metrics,
                    scheduler_class=scheduler_class,  
                    scheduler_kwargs=scheduler_kwargs,
                    scale_target=True,
                    impute_only_missing=False)
    ########################################
    # logging options                      #
    ########################################
    exp_logger = TensorBoardLogger(save_dir=f'logs/generation/grin/',
                                       name='tensorboard')

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=50,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs/prediction/grin/',
        filename='best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(
        max_epochs=500,
        default_root_dir='logs/prediction/grin/',
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=5,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(imputer, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    res_test = trainer.test(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('mask', None))
    res_functional = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
                          test_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
                          test_mape=numpy_metrics.mape(y_hat, y_true, mask))

    res_val = trainer.validate(imputer, datamodule=dm)
    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('mask', None))
    res_functional.update(
        dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
             val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
             val_mape=numpy_metrics.mape(y_hat, y_true, mask)))
    
    return res_val

if __name__ == '__main__':
    model_params = {
        'hidden_size': 32,
        'embedding_size': 16,
        'n_layers': 1,
        'kernel_size': 2,
        'decoder_order': 1,
        'layer_norm': True,
        'dropout': 0.05,
        'merge_mode': 'mean'
    }
    optim_params = {'lr': 0.00001, 'weight_decay': 0.01}
    
    optim = 'RMSprop' # SGD or Adam
    
    batch_size = 1
    
    res = run_imputation(model_params, optim, optim_params, batch_size)

    logger.info(res)