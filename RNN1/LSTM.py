from datasets import get_dateset, normalize, denormalize
import matplotlib.pyplot as plt
from Models import GTLSTM
import torch
import json
import os
# Magic

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = 'cpu'
torch.set_default_device(device)
DATASET_NAME = 'SynteticSin'
MODEL_NAME= 'model_GTLSTM'
# Model Parameters
hidden_size = 100
num_layers = 3
lr = 0.01
dropout = 0.2
bidirectional = True
debug = False
train_from_checkpoint = True

Train, Validation, Test = get_dateset(DATASET_NAME)

train_data = torch.Tensor(Train)
train_label = denormalize(train_data)
validation_data = torch.Tensor(Validation)

input_size = train_data.shape[-1]
output_size = input_size
num_time_steps = len(train_data)

model = GTLSTM(input_size, output_size, hidden_size, dropout, num_layers, bidirectional, 'L1Loss', lr, ['EarlyStopping'], device)

configs = input_size, output_size, hidden_size, dropout, num_layers, bidirectional, 'L1Loss', lr, ['EarlyStopping']
if not os.path.exists(f'./models/{MODEL_NAME}'):
    model = model.train_step(train_data, 10, 50)
    torch.save(model.state_dict(), f'./models/{MODEL_NAME}')
    with open(f'./models/{MODEL_NAME}.config', 'w') as config: 
        json.dump(configs, config)
else:
    state_dict = torch.load(f'./models/{MODEL_NAME}')
    model.load_state_dict(state_dict)
  
output = model.predict_step(train_data, start=0, steps=100)

print(output)
data_true = denormalize(name=DATASET_NAME, x=train_data[0:100, :].numpy())
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
    plt.savefig(f'./PNG/{DATASET_NAME}/{MODEL_NAME}_Feature_{i}.png')
    plt.clf()