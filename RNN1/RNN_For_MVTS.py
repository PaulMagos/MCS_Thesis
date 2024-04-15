from datasets import get_dateset
import matplotlib.pyplot as plt
from GTC import GTC
from GMM import gmm_loss
import numpy as np
import torch
import time
from tqdm import tqdm
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = 'cpu'
torch.set_default_device(device)

# Model Parameters
hidden_size = 200
num_layers = 5
lr = 0.001
dropout = 0.2
bidirectional = True
mixture_dim = 10
debug = False

EEGTrain, EEGValidation, EEGTest = get_dateset('EEG')
# EnergyTrain, EnergyValidation, EnergyTest = get_dateset('Energy')


train_data = torch.Tensor(EEGTrain)
validation_data = torch.Tensor(EEGValidation)

input_size = train_data.shape[-1]
output_size = input_size
num_time_steps = len(train_data)

args = (input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, device, debug)
model = GTC(*args)

criterion = gmm_loss
optimizer = optim.RMSprop(model.parameters(), lr=lr)

print(model)

# Training loop
print("Starting training...")
for epoch in range(1, 5 + 1):
    losses_epoch = []
    with tqdm(total=len(train_data) - 1) as pbar:
        for i in range(len(train_data) - 1):
            optimizer.zero_grad()
            outputs = model(train_data[i:i+1, :].to(device))
            loss = criterion(train_data[i+1, :].to(device), outputs)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            losses_epoch.append(loss.item())
            if i%200 == 0:
                mean_loss = np.mean(losses_epoch)
                pbar.set_description("Loss %s" % mean_loss)
            pbar.update(1)
    mean_loss = np.mean(losses_epoch)
    print(f'Epoch {epoch} - loss:', mean_loss)
    
start = time.time()
losses_epoch = []
with tqdm(total=100) as p:
    for i in range(1, 101): 
        outputs = model(train_data[i:i+1, :].to(device))
        loss = criterion(train_data[i+1, :].to(device), outputs)
        losses_epoch.append(loss.item())
        p.update(1)
print(f'Took {time.time()-start} on {device}')
mean_loss = np.mean(losses_epoch)
print('- loss:', mean_loss)

torch.save(model.state_dict(), './model')
