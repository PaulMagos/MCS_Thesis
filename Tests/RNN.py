from datasets import get_dateset
import matplotlib.pyplot as plt
from Models import GTR
import torch
import os
import json
# Magic

MODELS_PATH = f'{os.path.dirname(__file__)}/../models'
IMAGES_PATH = f'{os.path.dirname(__file__)}/../PNG'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_default_device(DEVICE)

DATASET_NAME = 'SynteticSin'
MODEL_NAME= 'GTR'

# Model Parameters
hidden_size = 1024
num_layers = 1
lr = 0.1
dropout = 0
bidirectional = True
debug = False
train_from_checkpoint = False

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

model = GTR(input_size, output_size, hidden_size, dropout, num_layers, bidirectional, lr, ['EarlyStopping'], DEVICE)
configs = input_size, output_size, hidden_size, dropout, num_layers, bidirectional, 'mse', lr, ['EarlyStopping']

try:
    state_dict = torch.load(f'{MODELS_PATH}/{MODEL_NAME}_{DATASET_NAME}')
    model.load_state_dict(state_dict)
except:
    print('Model not present or incompatible')
    train_from_checkpoint = True
    
if train_from_checkpoint:
    model, history = model.train_step(train_data, train_label, 1, 100, 100)
    torch.save(model.state_dict(), f'{MODELS_PATH}/{MODEL_NAME}_{DATASET_NAME}')
    with open(f'{MODELS_PATH}/{MODEL_NAME}.hist', 'w') as hist:
        json.dump(history, hist)
    with open(f'{MODELS_PATH}/{MODEL_NAME}.config', 'w') as config: 
        json.dump(configs, config)

output = model.predict_step(train_data, start=0, steps=100)

data_true = train_label[:100, :, :].numpy()
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
    plt.title(f'Line Plot of Feature {i}')
    plt.legend()
    plt.savefig(f'{IMAGES_PATH}/{DATASET_NAME}/{MODEL_NAME}_Feature_{i}.png')
    plt.clf()
    
output = model.generate_step(train_data, start=0, steps=100)

data_true = train_label[:100, :, :].numpy()
data_predicted = output.reshape(output.shape[0], output.shape[-1])
data_true = data_true.reshape(data_true.shape[0], data_true.shape[-1])
print(data_predicted.shape, data_true.shape)
for i in range(data_true.shape[-1]):
    first_elements_arr1 = [subarr[i] for subarr in data_true]
    first_elements_arr2 = [subarr[i] for subarr in data_predicted]
    # Plotting
    plt.plot(first_elements_arr1, label='True')
    plt.plot(first_elements_arr2, label='Generated')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title(f'Line Plot of Feature {i}')
    plt.legend()
    plt.savefig(f'{IMAGES_PATH}/{DATASET_NAME}/{MODEL_NAME}_Feature_{i}_GEN.png')
    plt.clf()
    
with open(f'{MODELS_PATH}/{MODEL_NAME}.hist', 'r') as hist:
    history = json.load(hist)
    
for key, values in history.items():
    plt.plot(values, label=key)
plt.savefig(f'{IMAGES_PATH}/{DATASET_NAME}/{MODEL_NAME}_History.png')
plt.clf()