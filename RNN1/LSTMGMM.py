from datasets import get_dateset, denormalize
import matplotlib.pyplot as plt
from Models import GTM
from GMM import gmm_loss
import json
import torch
import numpy as np
import os

DATASET_NAME = 'SynteticSin'

MODEL_NAME= 'model_LSTMGMM'
# Magic
# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'
torch.set_default_device(device)

# Model Parameters 100 hidden
hidden_size = 256
num_layers = 2
lr = 0.0001
weight_decay = 0.001
dropout = 0.2
stds_to_use = 10
bidirectional = False
mixture_dim = 20
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

train_data = torch.Tensor(Train)
train_label = train_data
train_data = train_data[:-1]
train_label = train_label[1:]

validation_data = torch.Tensor(Validation)
validation_label = validation_data
validation_data = validation_data[:-1]
validation_label = validation_label[1:]

input_size = train_data.shape[-1]
output_size = input_size
num_time_steps = len(train_data)

model = GTM(input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, gmm_loss, lr, weight_decay, ['EarlyStopping'], device, debug)

configs = input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, lr, weight_decay, ['EarlyStopping'], device, debug
try:
    state_dict = torch.load(f'./models/{MODEL_NAME}_{DATASET_NAME}')
    model.load_state_dict(state_dict)
except:
    print('Model not present or incompatible')
    train_from_checkpoint = True
    
if train_from_checkpoint:
    model = model.train_step(train_data, train_label, 1, 50, 100)
    torch.save(model.state_dict(), f'./models/{MODEL_NAME}_{DATASET_NAME}')
    with open(f'./models/{MODEL_NAME}.config', 'w') as config: 
        json.dump(configs, config)
  
output = model.predict_step(validation_data, start=0, steps=100)

data_true = validation_label[0:100, :, :].numpy()
data_predicted = output.reshape(output.shape[0], output.shape[-1])
data_true = data_true.reshape(data_true.shape[0], data_true.shape[-1])
print(data_predicted.shape, data_true.shape)
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
    plt.savefig(f'./PNG/{DATASET_NAME}/{MODEL_NAME}_Feature_{i}.png')
    plt.clf()