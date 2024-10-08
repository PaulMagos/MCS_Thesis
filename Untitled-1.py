# %%
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler, MinMaxScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from GRGN.Engines.Generator import Generator
from tsl.data import TemporalSplitter
import pandas as pd
from GRGN.Loss.LogLikelihood import LogLikelihood
from pytorch_lightning import Trainer
from tsl.utils.casting import torch_to_numpy
import matplotlib.pyplot as plt
from typing import Optional
from torch.utils.data import DataLoader

path = '/storagenfs/p.magos/TSGen/RES/PemsBay/Model2'

class CustomSpatioTemporalDataModule(SpatioTemporalDataModule):
    def train_dataloader(self, shuffle: bool = False,
                         batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('train', False, batch_size)

# %%
# filename = '/data/p.magos/TSGen/logs/generation/AirQuality/Best.ckpt'
filename = '/storagenfs/p.magos/TSGen/logs/generation/PemsBay/PemsBay2val_loss=-0.1397.ckpt'
# %%
filename

# %%
# dataset = AirQuality(impute_nans=True, small=True)
# dataset = MetrLA()
dataset = PemsBay()

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
torch_dataset = SpatioTemporalDataset(target=dataset.dataframe()[-5000:],
                                    covariates=None,
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
y_hat, y_true = (output['y_hat'], output['y'])
test_true = scalers['target'].transform(torch.tensor(y_true))
res = dict(test_mae=loss_fn.loss_function(torch.tensor(y_hat), torch.tensor(test_true)))
res['test_mae']

# %%
y_true = torch.Tensor(y_true)

# %%
kwargs = {'scaler': scalers['target']}
input = y_true[-599:-598]
generation = generator.autoregression(X=input, edge_index=torch.tensor(adj[0]), edge_weight=torch.Tensor(adj[1]), steps=1000, enc_dec_mean=True, **kwargs)
generation1 = generator.autoregression(X=input, edge_index=torch.tensor(adj[0]), edge_weight=torch.Tensor(adj[1]), steps=1000, enc_dec_mean=False, **kwargs)
generation2 = generator.generate(edge_index=torch.tensor(adj[0]), edge_weight=torch.Tensor(adj[1]), steps=1000, enc_dec_mean=True, **kwargs)
generation3 = generator.generate(edge_index=torch.tensor(adj[0]), edge_weight=torch.Tensor(adj[1]), steps=1000, enc_dec_mean=False, **kwargs)

generation_self = generator.autoregression(X=input, edge_index=torch.tensor(adj_self[0]), edge_weight=torch.Tensor(adj_self[1]), steps=1000, enc_dec_mean=True, **kwargs)
generation1_self = generator.autoregression(X=input, edge_index=torch.tensor(adj_self[0]), edge_weight=torch.Tensor(adj_self[1]), steps=1000, enc_dec_mean=False, **kwargs)
generation2_self = generator.generate(edge_index=torch.tensor(adj_self[0]), edge_weight=torch.Tensor(adj_self[1]), steps=1000, enc_dec_mean=True, **kwargs)
generation3_self = generator.generate(edge_index=torch.tensor(adj_self[0]), edge_weight=torch.Tensor(adj_self[1]), steps=1000, enc_dec_mean=False, **kwargs)

# %%
# true = y_true.reshape(y_true.shape[0], y_true.shape[-2])
generation = generation.reshape(generation.shape[1], generation.shape[-2])
generation1 = generation1.reshape(generation1.shape[1], generation1.shape[-2])
generation2 = generation2.reshape(generation2.shape[1], generation2.shape[-2])
generation3 = generation3.reshape(generation3.shape[1], generation3.shape[-2])

generation_self = generation_self.reshape(generation_self.shape[1], generation_self.shape[-2])
generation1_self = generation1_self.reshape(generation1_self.shape[1], generation1_self.shape[-2])
generation2_self = generation2_self.reshape(generation2_self.shape[1], generation2_self.shape[-2])
generation3_self = generation3_self.reshape(generation3_self.shape[1], generation3_self.shape[-2])

# %%
# plt.plot(generation[-99:, 35], label='Generated')
# plt.plot(generation1[-99:, 35], label='Generated1')
# plt.plot(generation2[-99:, 35], label='Generated1')
# plt.plot(generation3[-99:, 35], label='Generated1')
# plt.plot(true[-99:, 35], label='True')
# plt.legend()
# %%
# input = y_true[-200:-100].permute(1, 0, 2, 3)
# prediction1 = generator.predict(input, torch.tensor(adj[0]), torch.Tensor(adj[1]), steps=500, enc_dec_mean=True, **kwargs)
# prediction = generator.predict(input, torch.tensor(adj[0]), torch.Tensor(adj[1]), steps=500, enc_dec_mean=False, **kwargs)
# prediction1_self = generator.predict(input, torch.tensor(adj_self[0]), torch.Tensor(adj_self[1]), steps=500, enc_dec_mean=True, **kwargs)
# prediction_self = generator.predict(input, torch.tensor(adj_self[0]), torch.Tensor(adj_self[1]), steps=500, enc_dec_mean=False, **kwargs)

# %%
# prediction = prediction.reshape(prediction.shape[0], prediction.shape[-2])
# prediction1 = prediction1.reshape(prediction1.shape[0], prediction1.shape[-2])
# prediction_self = prediction_self.reshape(prediction_self.shape[0], prediction_self.shape[-2])
# prediction1_self = prediction1_self.reshape(prediction1_self.shape[0], prediction1_self.shape[-2])
# plt.plot(prediction[-500:-400, 34], label='Predicted')
# plt.plot(true[-100:, 34], label='True')
# plt.legend()

# %%
# X = true
# Y = prediction

# %%
# err = torch.square(prediction[-50:] - true[-50:])
# err.mean()

# %%
cols = dataset.dataframe().columns.droplevel('channels')
df = pd.DataFrame(generation, columns=cols)
df1 = pd.DataFrame(generation1, columns=cols)
df4 = pd.DataFrame(generation2, columns=cols)
df5 = pd.DataFrame(generation3, columns=cols)

df_self = pd.DataFrame(generation_self, columns=cols)
df1_self = pd.DataFrame(generation1_self, columns=cols)
df4_self = pd.DataFrame(generation2_self, columns=cols)
df5_self = pd.DataFrame(generation3_self, columns=cols)

# df2 = pd.DataFrame(prediction1, columns=cols)
# df2_self = pd.DataFrame(prediction1_self, columns=cols)
# df3 = pd.DataFrame(prediction, columns=cols)
# df3_self = pd.DataFrame(prediction_self, columns=cols)

# %%
df.to_csv(f'{path}/AutoregPemsBayGRGN.csv', index=False)
df1.to_csv(f'{path}/AutoregPemsBayGRGNnoENC.csv', index=False)
df4.to_csv(f'{path}/SamplingPemsBayGRGN.csv', index=False)
df5.to_csv(f'{path}/SamplingPemsBayGRGNnoENC.csv', index=False)

df_self.to_csv(f'{path}/Self/AutoregPemsBayGRGN.csv', index=False)
df1_self.to_csv(f'{path}/Self/AutoregPemsBayGRGNnoENC.csv', index=False)
df4_self.to_csv(f'{path}/Self/SamplingPemsBayGRGN.csv', index=False)
df5_self.to_csv(f'{path}/Self/SamplingPemsBayGRGNnoENC.csv', index=False)

# df2.to_csv(f'{path}/PredictMetrLAGRGN.csv', index=False)
# df2_self.to_csv(f'{path}/Self/PredictMetrLAGRGN.csv', index=False)
# df3.to_csv(f'{path}/PredictMetrLAGRGNnoENC.csv', index=False)
# df3_self.to_csv(f'{path}/Self/PredictMetrLAGRGNnoENC.csv', index=False)

# %%



