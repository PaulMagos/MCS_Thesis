from datasets import get_dateset, inverse_transform
import matplotlib.pyplot as plt
from GT import GTLSTM
import torch
import os
# Magic

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = 'cpu'
torch.set_default_device(device)

# Model Parameters
hidden_size = 2000
num_layers = 5
lr = 0.1
dropout = 0.2
bidirectional = True
debug = False
train_from_checkpoint = True

EEGTrain, EEGValidation, EEGTest = get_dateset('EEG')
# EnergyTrain, EnergyValidation, EnergyTest = get_dateset('Energy')


train_data = torch.Tensor(EEGTrain)
validation_data = torch.Tensor(EEGValidation)

input_size = train_data.shape[-1]
output_size = input_size
num_time_steps = len(train_data)

model = GTLSTM(input_size, output_size, hidden_size, dropout, num_layers, bidirectional, 'L1Loss', lr, ['EarlyStopping'], device)

if not os.path.exists('./model_GTLSTM'):
    model = model.train_step(train_data, 10)
    torch.save(model.state_dict(), './model_GTLSTM')
else:
    state_dict = torch.load('./model_GTLSTM')
    model.load_state_dict(state_dict)
    if train_from_checkpoint:
        model = model.train_step(train_data, 10)
        torch.save(model.state_dict(), './model_GTLSTM')

output = model.predict_step(train_data, start=100, steps=7)

data_true = inverse_transform(train_data[:7, :])
data_predicted = inverse_transform(output)

first_elements_arr1 = [subarr[0] for subarr in data_true]
first_elements_arr2 = [subarr[0] for subarr in data_predicted]
# Plotting
plt.plot(first_elements_arr1, label='True')
plt.plot(first_elements_arr2, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Line Plot of First Arrays')
plt.legend()
plt.savefig('GTLSTM.png')
