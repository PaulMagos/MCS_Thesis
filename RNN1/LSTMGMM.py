from datasets import get_dateset, denormalize
import matplotlib.pyplot as plt
from Models import GTM
from GMM import gmm_loss
import torch
import numpy as np
import os

# Magic

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'
torch.set_default_device(device)

# Model Parameters
hidden_size = 100
num_layers = 5
lr = 0.001
dropout = 0.2
stds_to_use = 10
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

model = GTM(input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, gmm_loss, lr, ['EarlyStopping'], device, debug)


if not os.path.exists('./model_GTM'):
    model = model.train_step(train_data, 10)
    torch.save(model.state_dict(), './models/model_LSTMGMM')
else:
    state_dict = torch.load('./models/model_LSTMGMM')
    model.load_state_dict(state_dict)
  
# output = model.predict_step(train_data, start=0, steps=1000)

# data_true = inverse_transform(train_data[:1000, :])
# data_predicted = inverse_transform(output)

# first_elements_arr1 = [subarr[0] for subarr in data_true]
# first_elements_arr2 = [subarr[0] for subarr in data_predicted]
# # Plotting
# plt.plot(first_elements_arr1, label='True')
# plt.plot(first_elements_arr2, label='Predicted')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title('Line Plot of First Arrays')
# plt.legend()
# plt.savefig('LSTMGMM.png')