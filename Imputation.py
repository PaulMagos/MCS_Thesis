# %%
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler, MinMaxScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from GRGN.Engines.Generator import Generator
from tsl.data import TemporalSplitter
from tsl.ops.imputation import add_missing_values
import pandas as pd
from tsl.transforms import MaskInput
from GRGN.Loss.LogLikelihood import LogLikelihood
from pytorch_lightning import Trainer
from tsl.utils.casting import torch_to_numpy
import matplotlib.pyplot as plt
from typing import Optional
from torch.utils.data import DataLoader

path = '/storagenfs/p.magos/TSGen/RES/AirQuality/Model4/'

class CustomSpatioTemporalDataModule(SpatioTemporalDataModule):
    def train_dataloader(self, shuffle: bool = False,
                         batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('train', False, batch_size)

# %%
# filename = '/data/p.magos/TSGen/logs/generation/AirQuality/Best.ckpt'
filename = '/storagenfs/p.magos/TSGen/logs/generation/AirQuality/AirQuality_LargeHSmallM-model-epoch=57-val_loss=-0.8160.ckpt'

# %%
filename

# %%
dataset = AirQuality(impute_nans=True, small=True)
p_fault, p_noise = 0., 0.25

# dataset = add_missing_values(PemsBay(),
#                                 p_fault=p_fault,
#                                 p_noise=p_noise,
#                                 min_seq=12,
#                                 max_seq=12 * 4,
#                                 seed=56789)
# 
# dataset = add_missing_values(MetrLA(),
#                         p_fault=p_fault,
#                         p_noise=p_noise,
#                         min_seq=12,
#                         max_seq=12 * 4,
#                         seed=9101112)

adj = dataset.get_connectivity(**{'method': 'distance',
'threshold': 0.1,
'include_self': False,
'layout': 'edge_index'
})

adj_self = dataset.get_connectivity(**{'method': 'distance',
'threshold': 0.1,
'include_self': True,
'layout': 'edge_index'
})

# instantiate dataset
torch_dataset = ImputationDataset(target=dataset.dataframe()[-5000:],
                                    mask=dataset.training_mask[-5000:],
                                    eval_mask=dataset.eval_mask[-5000:],
                                    covariates=None,
                                    transform=MaskInput(),
                                    connectivity=adj,
                                    window=1,
                                    stride=1)
splitter = TemporalSplitter(0.1, 0.2)
scalers = {'target': StandardScaler(axis=(0, 1))}
dm = CustomSpatioTemporalDataModule(
    dataset=torch_dataset,
    splitter=splitter,
    scalers=scalers,
    batch_size=1,
    workers=8)
dm.setup(stage='test') 

# if cfg.get('in_sample', False):
dm.trainset = list(range(len(torch_dataset)))

# %%
# setup generator
generator = Generator.load_from_checkpoint(filename)

# %%
trainer = Trainer()

trainer.ckpt_path= filename

generator.freeze()
# trainer.test(generator, datamodule=dm)

# %%
loss_fn = LogLikelihood()
generator.hparams
# %%
output = trainer.predict(generator, dataloaders=dm.test_dataloader())
output = generator.collate_prediction_outputs(output)
output = torch_to_numpy(output)
y_hat, y_true, mask = (output['y_hat'], output['y'], output['mask'])
test_true = scalers['target'].transform(torch.tensor(y_true))
mask = torch.tensor(mask).bool()
res = dict(test_mae=loss_fn.loss_function(torch.tensor(y_hat), torch.tensor(test_true)))
res['test_mae']

# %%
y_true = torch.Tensor(y_true)

# %%
kwargs = {'scaler': scalers['target']}
input = y_true[-500:]
mask_ = mask[-500:]
# imputation = generator.imputation(X=input, mask=mask_, edge_index=torch.tensor(adj[0]), edge_weight=torch.Tensor(adj[1]), enc_dec_mean=True, **kwargs)
# imputation1 = generator.imputation(X=input, mask=mask_, edge_index=torch.tensor(adj[0]), edge_weight=torch.Tensor(adj[1]), enc_dec_mean=False, **kwargs)
imputation_self = generator.imputation(X=input, mask=mask_, edge_index=torch.tensor(adj_self[0]), edge_weight=torch.Tensor(adj_self[1]), enc_dec_mean=True, **kwargs)
# imputation1_self = generator.imputation(X=input, mask=mask_, edge_index=torch.tensor(adj_self[0]), edge_weight=torch.Tensor(adj_self[1]), enc_dec_mean=False, **kwargs)
# %%
true = y_true.reshape(y_true.shape[0], y_true.shape[-2])
# imputation = imputation.reshape(imputation.shape[0], imputation.shape[-2])
# imputation1 = imputation1.reshape(imputation1.shape[0], imputation1.shape[-2])
imputation_self = imputation_self.reshape(imputation_self.shape[0], imputation_self.shape[-2])
# imputation1_self = imputation1_self.reshape(imputation1_self.shape[0], imputation1_self.shape[-2])

# %%
cols = dataset.dataframe().columns.droplevel('channels')
# df = pd.DataFrame(imputation, columns=cols)
# df1 = pd.DataFrame(imputation1, columns=cols)
df_self = pd.DataFrame(imputation_self, columns=cols)
# df1_self = pd.DataFrame(imputation1_self, columns=cols)

# %%
# df.to_csv(f'{path}/ImputationMetrLAGRGN.csv', index=False)
# df1.to_csv(f'{path}/ImputationMetrLAGRGNnoENC.csv', index=False)
df_self.to_csv(f'{path}/Self/ImputationAirQualityGRGN.csv', index=False)
# df1_self.to_csv(f'{path}/Self/ImputationMetrLAGRGNnoENC.csv', index=False)
# %%



