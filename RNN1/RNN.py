import torch
import torch.nn as nn
from datasets import get_dateset, inverse_transform
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_default_device(device)

EEGTrain, EEGValidation, EEGTest = get_dateset('EEG')


train_data = torch.Tensor(EEGTrain)[:3010]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, num_layers):
        super(RNN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, device=self.device, num_layers=num_layers)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size, device=self.device)

    def forward(self, x, hidden):
        x = x.to(self.device)
        # Forward pass through the RNN layer
        out, hidden = self.rnn(x, hidden)
        
        # Reshape the output to be (batch_size * seq_length, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size)
        
        # Forward pass through the output layer
        out = self.fc(out)
        
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

# Example usage:
input_size = 14  # Input size (e.g., number of features in your time series)
hidden_size = 100  # Hidden size of the RNN layer
output_size = 1  # Output size (e.g., number of predicted values)
num_layers = 10
# Create an instance of the RNN model
model = RNN(input_size, hidden_size, output_size, device, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Example training loop (you should adapt this to your data)
# Assuming you have input_data and target_data as your input and target time series
# input_data and target_data should be torch Tensors
num_epochs = 1000
batch_size = 1

# Define sequence length
sequence_length = 10  # For example, using a sequence length of 10

# Create input-output pairs
input_sequences = []
target_values = []
for i in range(len(train_data) - sequence_length):
    input_seq = train_data[i:i+sequence_length]
    target_val = train_data[i+sequence_length]
    input_sequences.append(input_seq)
    target_values.append(target_val)
    
    
input_sequences = torch.stack(input_sequences)
target_values = torch.stack(target_values)

i = 0
for epoch in range(num_epochs):
    model.train()
    hidden = model.init_hidden(batch_size)
    
    # Forward pass
    outputs, hidden = model(input_sequences[i:i+1], hidden)
    loss = criterion(outputs, target_values.view(-1).to(device))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
