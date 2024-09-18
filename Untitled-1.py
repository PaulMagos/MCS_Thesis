# %%
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality
from GRGN.Engines.Generator import Generator
from tsl.data import TemporalSplitter
import pandas as pd
from GRGN.Loss.LogLikelihood import LogLikelihood
from pytorch_lightning import Trainer
from tsl.utils.casting import torch_to_numpy
import matplotlib.pyplot as plt
from typing import Optional
from torch.utils.data import DataLoader

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

adj = dataset.get_connectivity(**{'method': 'distance',
'threshold': 0.1,
'include_self': False,
'layout': 'edge_index'
})

# instantiate dataset
torch_dataset = SpatioTemporalDataset(target=dataset.dataframe()[-1100:],
                                    covariates=None,
                                    connectivity=adj,
                                    window=1,
                                    stride=1)
splitter = TemporalSplitter(0.5, 0)
scalers = {'target': StandardScaler(axis=(0, 1))}
dm = CustomSpatioTemporalDataModule(
    dataset=torch_dataset,
    splitter=splitter,
    scalers=scalers,
    batch_size=1,
    workers=8)
dm.setup(stage='fit') 

# if cfg.get('in_sample', False):
dm.trainset = list(range(len(torch_dataset)))

# %%
scalers['target'].params

# %%
# setup generator
generator = Generator.load_from_checkpoint(filename)

# %%
trainer = Trainer()

trainer.ckpt_path= filename

generator.freeze()
# trainer.test(generator, datamodule=dm)

# %%
print(generator.hparams)
loss_fn = LogLikelihood()

# %%
output = trainer.predict(generator, dataloaders=dm.train_dataloader())
output = generator.collate_prediction_outputs(output)
output = torch_to_numpy(output)
y_hat, y_true = (output['y_hat'], output['y'])
test_true = scalers['target'].transform(torch.tensor(y_true))
res = dict(test_mae=loss_fn.loss_function(torch.tensor(y_hat), torch.tensor(test_true)))
res['test_mae']/2

# %%
y_true = torch.Tensor(y_true)
y_true.shape

# %%
kwargs = {'scaler': scalers['target']}
input = y_true[-1000:-999]
generation = generator.generate(input, torch.tensor(adj[0]), torch.Tensor(adj[1]), None, 1000, enc_dec_mean=False, **kwargs)

# %%
print(generation.shape)
true = y_true.reshape(y_true.shape[0], y_true.shape[-2])
generation = generation.reshape(generation.shape[0], generation.shape[-2])

# %%
X = true[:]
Y = generation

# %%
# plt.plot(generation[-500:-400, 35], label='Generated')
# plt.plot(true[-499:-399, 35], label='True')

plt.plot(generation[-99:, 1], label='Generated')
plt.plot(true[-99:, 1], label='True')


plt.legend()

# %%
y_true.shape

input = y_true[-100:]
# input = scalers['target'].transform(input)
input.shape
prediction = generator.predict(input, torch.tensor(adj[0]), torch.Tensor(adj[1]), enc_dec_mean=True, **kwargs)

# %%
prediction = prediction.reshape(prediction.shape[0], prediction.shape[-2])
prediction.shape

# %%
# plt.plot(prediction[-100:, 1], label='Predicted')
# plt.plot(true[-100:, 1], label='True')

plt.plot(prediction[-100:, 34], label='Predicted')
plt.plot(true[-100:, 34], label='True')


# plt.plot(prediction[:, 100], label='Predicted')
# plt.plot(true[-99:, 100], label='True')
plt.legend()

# %%
X = true
Y = prediction

# %%
err = torch.square(prediction[-99:] - true[-99:])
err.mean()

# %%
cols = [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036]
df = pd.DataFrame(generation, columns=cols)

# %%
df.to_csv('SynteticAirQualityGRGN.csv', index=False)

# %%



