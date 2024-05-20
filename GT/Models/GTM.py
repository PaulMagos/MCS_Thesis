from torch import nn
from .GMM import GMM
import torch.optim as optim
from .EarlyStopping import EarlyStopping
import numpy as np
import torch
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

__all__ = ['GTM']

class GTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, loss, lr, weight_decay, callbacks, device, debug) -> None:
        super(GTM, self).__init__()
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.activation1 = nn.Tanh()
        
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=(output_size+2)*mixture_dim, device=device)
        self.gmm = GMM(M = mixture_dim, device = device, debug=debug)
        
        self.loss = loss
        self.device = device
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()

    def forward(self, x):
        out = self.lstm(x)[0]
        activation1 = self.activation1(out)
        activation1 = out
        dense = self.dense(activation1)
        gmm = self.gmm(dense)
        out_activation = gmm
        return out_activation

    def train_step(self, train_data, train_label=None, epochs = 1, step_log=500):
        if train_label==None:
            train_label = train_data[1:]
            train_data = train_data[:-1]
        train_label = torch.Tensor(train_label).type(torch.float32)
        train_data = train_data.to(self.device)
        train_label = train_label.to(self.device)
        self.train()
        print("Starting training...")
        history = {'loss': []}
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            with tqdm(total=len(train_data)) as pbar:
                for i in range(len(train_data)):
                    self.optimizer.zero_grad()
                    outputs = self(train_data[i:i+1, :, :])
                    loss = self.loss(train_label[i:i+1, :, :], outputs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    if i%step_log == 0:
                        mean_loss = np.mean(losses_epoch)
                        pbar.set_description(f"Loss {mean_loss}")
                    pbar.update(1)

            mean_loss = np.mean(losses_epoch)
            history['loss'].append(mean_loss)
            print(f'Epoch {epoch} - loss:', mean_loss)
            if self.callbacks['EarlyStopping'](self, mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break

        return self, history

    def predict_step(self, data, start = 0, steps = 7, mode='mean'):
        M = self.gmm.M
        D = data.shape[-1]
        self.eval()
        output = torch.Tensor()
        data = data.to(self.device)
        with tqdm(total=steps) as pbar:
            for i in range(start, start+steps):
                data_ = self(data[i:i+1, :, :])
                means = data_[:, :M*D]
                stds = data_[:, M*D : M*(D+1)]
                gmm_weights = data_[:, M*(D+1):]
                
                means = means.reshape(-1, M, D)
                stds = stds.unsqueeze(-1)
                gmm_weights = gmm_weights.unsqueeze(-1)
                
                pred = gmm_weights * torch.normal(means, stds)
                match(mode):
                    case 'mean':
                        pred = torch.mean(pred, axis=1)
                    case 'sum':
                        pred = torch.sum(pred, axis=1)
                output = torch.concat([output, pred])
                pbar.update(1)
        
        return np.array(output.cpu().detach())

    def generate_step(self, data, start = 0, steps = 7, mode='mean'):
        M = self.gmm.M
        D = data.shape[-1]
        self.eval()
        output = torch.Tensor(data[start:start+1, :, :])
        input = output.to(self.device)
        output = output.reshape(1, -1)
        with tqdm(total=steps) as pbar:
            for i in range(start, start+steps):
                data_ = self(input)
                means = data_[:, :M*D]
                stds = data_[:, M*D : M*(D+1)]
                gmm_weights = data_[:, M*(D+1):]
                
                means = means.reshape(-1, M, D)
                stds = stds.unsqueeze(-1)
                gmm_weights = gmm_weights.unsqueeze(-1)
                
                pred = gmm_weights * torch.normal(means, stds)
                match(mode):
                    case 'mean':
                        pred = torch.mean(pred, axis=1)
                    case 'sum':
                        pred = torch.sum(pred, axis=1)
                input = pred.reshape(1, 1, -1)
                output = torch.concat([output, pred])
                pbar.update(1)
        return np.array(output.cpu().detach())
