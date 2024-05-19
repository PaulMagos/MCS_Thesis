from torch import nn
import torch.optim as optim
from Models.EarlyStopping import EarlyStopping
import numpy as np
import torch
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

__all__ = ['GTLSTM']

class GTLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, num_layers, bidirectional, loss, lr, callbacks, device) -> None:
        super(GTLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.activation1 = nn.Tanh()
        self.out_activation = nn.Tanh()
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=output_size, device=device)
        # self.Relu = nn.Linear(output_size, output_size)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.3)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()
        self.device = device
        match(loss):
            case 'L1Loss':
                self.loss = nn.L1Loss(reduction='mean')
                self.metrics = nn.MSELoss()
            case 'mse':
                self.loss = nn.MSELoss()
                self.metrics = nn.L1Loss(reduction='mean')

    def forward(self, x):
        out = self.lstm(x)[0]
        activation1 = self.activation1(out)
        dense = self.dense(activation1)
        out_activation = self.out_activation(dense)
        return out_activation

    def train_step(self, train_data, train_label=None, batch_size=1, epochs = 1, step_log=500):
        self.train()
        if train_label==None:
            train_label = train_data[1:]
            train_data = train_data[:-1]
        train_label = torch.Tensor(train_label).type(torch.float32)
        print("Starting training...")
        history = {'loss': [], 'MAE': []}
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            metrics_epoch = []
            with tqdm(total=len(train_data)//batch_size+1) as pbar:
                for i in range(0, len(train_data), batch_size):
                    window =  i+batch_size-32 if (i+batch_size - 32 >0) else 0
                    self.optimizer.zero_grad()
                    outputs = self(train_data[window:i+batch_size, :, :].to(self.device))
                    loss = self.loss(train_label[window:i+batch_size, :, :].to(self.device), outputs)
                    metric = self.metrics(train_label[window:i+batch_size, :, :].to(self.device), outputs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20)
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    metrics_epoch.append(metric.item())
                    if i%step_log == 0:
                        mean_loss = np.mean(losses_epoch)
                        mean_metric = np.mean(metrics_epoch)
                        pbar.set_description(f"Loss {mean_loss}, MAE : {mean_metric}")
                    pbar.update(1)

            mean_loss = np.mean(losses_epoch)
            mean_metric = np.mean(metrics_epoch)
            history['loss'].append(mean_loss)
            history['MAE'].append(mean_metric)
            print(f'Epoch {epoch} - MSE: {mean_loss} - MAE: {mean_metric}')
            if self.callbacks['EarlyStopping'](self, mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break

        return self, history
    
    def predict_step(self, data, start = 0, steps = 7):
        self.eval()
        data_ = data[start:start+steps-1, :].to(self.device)
        tmp_out = self(data_).cpu().detach().numpy()
        return tmp_out
    
    def generate_step(self, data, start = 0, steps = 7):
        self.eval()
        output = data[start, :].to(self.device)
        for i in range(steps):
            window =  i-32 if (i - 32 >0) else 0
            tmp_out = self(output[window:])[-1, :].reshape(1, -1)
            output = torch.concatenate([output, tmp_out])
        return output.cpu().detach().numpy()
