import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader, random_split


np.random.seed(42)

# TODO
# Fix Dataset Class
# Change permutes in training/validation



class IsCookingDataset(Dataset):

    def __init__(self, PATH):

        self.recipe = pd.read_json(PATH)

    def __len__(self):
        return len(self.recipe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cousine = self.recipe.iloc[idx, 1]
        ingredients = self.recipe.iloc[idx, 2]

        ingredients = np.array([ingredients.split(',', expand=True)])
        # ingredients = ingredients.astype('float').reshape(-1, 2)

        sample = {
            'cousine': torch.tensor(cousine),
            'ingredients': torch.from_numpy(ingredients)}

        return sample


def train_val_split(dataset, batch_train, batch_val, val_ratio=.2, shuffle=True, seed=None):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_ratio * dataset_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_val, sampler=val_sampler)

    return train_loader, val_loader

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, batch_size, output_dim=20, regression=False):
        super(LSTM, self).__init__()

        self.regression = regression
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.output_dim = output_dim
        self.batch_size = batch_size

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # setup output layer
        self.linear_full = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.linear_half = nn.Linear(self.hidden_dim // 2, self.output_dim)

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def last_by_index(self, outputs, lengths):

        idx = (lengths - 1).view(-1, 1).expand(outputs.shape[1], outputs.shape[2]).unsqueeze(0)
        return outputs.gather(0, idx).squeeze()

    def forward(self, x, length):

        out, hidden = self.lstm(x)
        out = self.last_by_index(out, length)
        out = torch.tanh(self.linear_full(out))
        out = torch.tanh(self.linear_half(out))

        if self.regression:
            out = out.view(-1)
        else:
            out = nn.functional.log_softmax(out, dim=1)

        return out

    def save(self, file):
        torch.save({'net_state_dict': self.state_dict()}, file)
        return self


def lstm_train(model, loss_function, optimizer, n_epochs, X, batch_size=32):
    # All time validation losses
    validation_alltime = []

    # Means
    train_mean, val_mean = [], []

    for epoch in range(n_epochs):

        # Reset loaders
        train_loader, val_loader = train_val_split(X, batch_size, batch_size, val_ratio=0.3)

        # init weights with 0s
        model.hidden = model.init_hidden()
        model.train()

        # This epoch losses.
        train_losses, val_losses = [], []

        # Training
        for train_batch, train_labels, test_length in train_loader:
            # zero out gradient, so they don't accumulate btw epochs
            model.zero_grad()

            X_batch = train_batch.permute(1, 0, 2)

            # Train
            y_pred = model(X_batch.float(), test_length)

            loss = loss_function(y_pred, train_labels)
            loss.backward()
            optimizer.step()

            # Grab losses
            _loss = loss.detach().item()
            train_losses.append(_loss)

        # Validation
        with torch.no_grad():
            model.eval()

            # init weights with 0s
            model.hidden = model.init_hidden()
            for val_batch, val_labels, val_length in val_loader:

                X_batch = val_batch.permute(1, 0, 2)

                # Validate
                y_pred = model(X_batch.float(), val_length)

                val_loss = loss_function(y_pred, val_labels)

                # Grab losses
                _loss = val_loss.detach().item()
                val_losses.append(_loss)
                validation_alltime.append(_loss)

        train_losses, val_losses = np.asarray(train_losses).mean(), np.asarray(val_losses).mean()
        print(f'Epoch: {epoch}, Training Loss: {train_losses}, Validation Loss: {val_losses}.')

        train_mean.append(train_losses)
        val_mean.append(val_losses)

    plt.plot(train_mean, 'b-', label='Training')
    plt.plot(val_mean, 'r-', label='Validation')
    plt.legend()
    plt.show()

    return model


def lstm_predict(model, X):

    y_hats = []

    with torch.no_grad():
        model.eval()
        model.hidden = model.init_hidden()

        for test_batch, test_labels, test_length in X:

            X_batch = test_batch.permute(1, 0, 2)

            y_pred = model(X_batch.float(), test_length)
            y_hat = np.argmax(y_pred.numpy(), axis=1)

            y_hats.append(y_hat)

    y_hats = np.hstack(y_hats)

    return np.asarray(y_hats)
