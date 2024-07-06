import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from load_data import get_dataset

import torch.optim as optim
from time_series_transformer import TimeSeriesTransformer
from time_series_transformer_batchnorm import TSTransformerEncoder
from utils import get_dataloaders


ds_list = ["UniMiB SHAR",
           "UCI HAR",
           "TWristAR",
           "Leotta_2021",
           "Gesture Phase Segmentation"
           ]
for i in ds_list:
    dataset = i
    print("**** ", dataset, " ****")

X_train, y_train, X_valid, y_valid, X_test, y_test, k_size, EPOCHS, t_names = get_dataset(dataset)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.long))

train_loader, valid_loader, test_loader = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test)


input_timesteps = X_train.shape[1]
in_channels = X_train.shape[2]
patch_size = 16
embedding_dim = 128
num_classes = len(torch.unique(y_train))

# Choose positional encoding type
pos_encoding_type = 'fixed'

# Instantiate models with chosen positional encoding
model1 = TimeSeriesTransformer(input_timesteps, in_channels, patch_size, embedding_dim, pos_encoding=pos_encoding_type, num_classes=num_classes)
model2 = TSTransformerEncoder(feat_dim=in_channels, max_len=input_timesteps, d_model=embedding_dim, n_heads=8, num_layers=6, dim_feedforward=128, norm='BatchNorm', pos_encoding=pos_encoding_type, num_classes=num_classes)

# Set up optimizer and criterion
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training function
def train_model(model, optimizer, train_loader, valid_loader, num_epochs=10):
    best_acc = 0.0
    best_model_wts = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(valid_loader.dataset)
        val_acc = val_running_corrects.double() / len(valid_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model

# Train both models
num_epochs = 10
print("Training Model 1...")
best_model1 = train_model(model1, optimizer1, train_loader, valid_loader, num_epochs)

print("Training Model 2...")
best_model2 = train_model(model2, optimizer2, train_loader, valid_loader, num_epochs)

# Evaluate both models on test data
def evaluate_model(model, test_loader):
    model.eval()
    test_running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_running_corrects += torch.sum(preds == labels.data)

    test_acc = test_running_corrects.double() / len(test_loader.dataset)
    print(f'Test Acc: {test_acc:.4f}')
