from datasets import get_dateset, denormalize
import matplotlib.pyplot as plt
from Models import GTM
from GMM import gmm_loss
import json
import torch
import numpy as np
import os

DATASET_NAME = 'EEG'

MODEL_NAME= 'model_LSTMGMM'
# Magic
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = 'cpu'
torch.set_default_device(device)

# Model Parameters 100 hidden
hidden_size = 100
num_layers = 5
lr = 0.001
dropout = 0.2
stds_to_use = 10
bidirectional = True
mixture_dim = 10
debug = False

# # Model Parameters 500 hidden
# hidden_size = 500
# num_layers = 5
# lr = 0.005
# dropout = 0.2
# stds_to_use = 10
# bidirectional = True
# mixture_dim = 20
# debug = False

Train, Validation, Test = get_dateset(DATASET_NAME)
# EnergyTrain, EnergyValidation, EnergyTest = get_dateset('Energy')

train_data = torch.Tensor(Train)
validation_data = torch.Tensor(Validation)

input_size = train_data.shape[-1]
output_size = input_size
num_time_steps = len(train_data)

model = GTM(input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, gmm_loss, lr, ['EarlyStopping'], device, debug)

configs = input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, lr, ['EarlyStopping'], device, debug
if not os.path.exists(f'./models/{MODEL_NAME}'):
    model = model.train_step(train_data, 10)
    torch.save(model.state_dict(), f'./models/{MODEL_NAME}')
    with open(f'./models/{MODEL_NAME}.config', 'w') as config: 
        json.dump(configs, config)
else:
    state_dict = torch.load(f'./models/{MODEL_NAME}')
    model.load_state_dict(state_dict)
  
output = model.predict_step(train_data, start=1000, steps=7)

data_true = denormalize(name=DATASET_NAME, x=train_data[1000:1007, :].numpy())
data_predicted = denormalize(name=DATASET_NAME, x=output)

for i in range(data_true.shape[-1]):
    first_elements_arr1 = [subarr[i] for subarr in data_true]
    first_elements_arr2 = [subarr[i] for subarr in data_predicted]
    # Plotting
    plt.plot(first_elements_arr1, label='True')
    plt.plot(first_elements_arr2, label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Line Plot of First Arrays')
    plt.legend()
    plt.savefig(f'./PNG/{DATASET_NAME}/LSTMGMM_Feature_{i}.png')
    plt.clf()