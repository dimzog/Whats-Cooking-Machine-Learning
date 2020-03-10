import numpy as np
from zografunk import model_lstm as lstm
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)

dataset = lstm.IsCookingDataset(PATH='data/train.json')

train, val = lstm.train_val_split(dataset=dataset, batch_train=32, batch_val=32, val_ratio=0.3)

n_epochs = 1
batch_size = 32

model = lstm.LSTM(input_size=128, hidden_dim=5, batch_size=batch_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model = lstm.lstm_train(X=train, model=model, n_epochs=n_epochs, optimizer=optimizer, loss_function=loss_function, batch_size=batch_size)

print(dataset)