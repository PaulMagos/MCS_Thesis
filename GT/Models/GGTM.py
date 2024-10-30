from torch import nn
from .GMM import GMM
import torch.optim as optim
from .EarlyStopping import EarlyStopping
import numpy as np
import torch
from ts2vg import NaturalVG
from tsl.nn.layers.graph_convs import DiffConv
from tqdm import tqdm
import random
from tsl.ops.connectivity import adj_to_edge_index
from vector_vis_graph import WeightMethod, horizontal_vvg, natural_vvg

__all__ = ['GGTM']

class GGTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, lr, weight_decay, callbacks, device, exo_size=0) -> None:
        super(GGTM, self).__init__()
        self.lin_in = nn.Linear(input_size, hidden_size)
        self.diff_conv = DiffConv(hidden_size, hidden_size, 2, root_weight=False)   
        # LSTM Layer
        # self.lstm = nn.LSTM(input_size=input_size+hidden_size+exo_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional, batch_first=True)
        self.gru = nn.GRU(input_size=input_size+hidden_size+exo_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional, batch_first=True)
        self.gmm = GMM(mixture_dim, hidden_size*(2 if bidirectional else 1), output_size, device = device)
        self.input_size = input_size
        self.loss = GMM.get_loss()
        self.device = device
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()
            
        self.horizon = 1
        self.window = 1
    
    def get_adj(self, ts):
        nvvg_adj = torch.tensor(natural_vvg(torch.tensor(ts).numpy(), weight_method=WeightMethod.TIME_DIFF_EUCLIDEAN_DISTANCE, directed=True))
        edge_index, edge_weights = adj_to_edge_index(nvvg_adj)
        return edge_index, edge_weights.float()
                
    def forward(self, x, exo_var=None):
        edge_i, edge_w = self.get_adj(x[0])
        x_in = self.lin_in(x)
        diff_ = self.diff_conv(x_in, edge_i, edge_w)
        x = torch.cat([diff_, x], dim=-1)
        
        if exo_var is not None:
            x = torch.cat([exo_var, x], dim=-1)
            
        out = self.gru(x)[0]
        gmm = self.gmm(out)
        return gmm

    def train_step(self, train_data, exo_var=None, batch_size=32, window=1, horizon=1, epochs = 1):
        train_data = train_data.to(self.device)
        val_data = train_data

            
        self.train()
        print("Starting training...")
        history = {'loss': []}
        windows = (train_data.shape[1]-horizon) // window
        batches = len(train_data)
        
        self.horizon = horizon
        self.window = window
        
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            
            with tqdm(total=batches) as pbar:
                for batch in range(0, batches, batch_size):
                    for i in range(windows):
                        batch_idx = i
                        
                        inputs = train_data[batch:batch+batch_size, batch_idx:batch_idx+window, :]
                        check = val_data[batch:batch+batch_size, batch_idx+horizon:batch_idx+window+horizon, :]
                        exo = exo_var[batch:batch+batch_size, batch_idx:batch_idx+window] if exo_var is not None else None
                        
                        mu, sigma, pi = self(inputs, exo)
                        loss = self.loss(check, mu, sigma, pi)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        losses_epoch.append(loss.item())
                        mean_loss = np.mean(losses_epoch)
                    pbar.set_description(f"Loss {mean_loss}")
                    pbar.update(batch_size)

            mean_loss = np.mean(losses_epoch)
            history['loss'].append(mean_loss)
            print(f'Epoch {epoch} - loss:', mean_loss)
            if self.callbacks['EarlyStopping'](self, mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break
        return self, history

    def predict_step(self, data, exo_var=None, start = 0, steps = 7):
        M = self.gmm.M
        D = data.shape[-1]
        self.eval()
        output = torch.Tensor().to(self.device)
        data = data.to(self.device)
        
        with tqdm(total=steps) as pbar:
            for i in range(start, start+steps):
                inputs = data[:, :i+1]
                
                if exo_var is not None:
                    inputs = torch.cat([exo_var[:, :i+1], inputs], dim=-1) 
                
                mu, sigma, pi = self(inputs)
                
                pred = GMM.sample(mu, sigma, pi).to(self.device)
                
                output = torch.concat([output, pred[:, i:i+1]], axis=1)
                pbar.update(1)
        
        return np.array(output.cpu().detach())

    def generate_step(self, shape: tuple, exo_var=None, window=None, horizon=None):
        
        num_timeseries = shape[0]
        steps = shape[1]
        
        shape = shape
        
        if window is None:
            window = self.window
        if horizon is None:
            horizon = self.horizon
        
        input_shape = (1, window, shape[2])
        
        
        exo = exo_var[0:1, :window] if exo_var is not None else None
        
        mu, sigma, pi = self(torch.zeros(input_shape), exo)
        inputs = GMM.sample(mu, sigma, pi)
       
        self.eval()
            
        
        total_output = None
        
        with tqdm(total=num_timeseries, disable=True if num_timeseries == 1 else False) as pbarTS:
            for ou in range(num_timeseries):
                output = None
                with tqdm(total=(steps-1)//horizon, disable=True if num_timeseries > 1 else False) as pbar:
                    for i in range((steps-1)//horizon):
                        if exo_var is not None:
                            exo = exo_var.view(exo_var.shape[0]*exo_var.shape[1], 1)[i+1:i+1 + window].view(1, window, 1) if exo_var is not None else None
                        
                        mu, sigma, pi = self(inputs, exo)
                        
                        pred = GMM.sample(mu, sigma, pi).to(self.device)
                        
                        output = torch.concat([output, pred[:, -horizon:]], axis=1) if output is not None else pred[:, -horizon:]
                        inputs = pred[:, -window:]
                        pbar.update(1)
                total_output = torch.cat([total_output, output], dim=0) if total_output is not None else output
                pbarTS.update(1)
        return np.array(total_output.cpu().detach())
