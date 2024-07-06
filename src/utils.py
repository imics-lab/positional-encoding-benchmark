import torch

def get_dataloaders(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, batch_size=32):
    train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    valid_data = torch.utils.data.TensorDataset(torch.Tensor(X_valid), torch.Tensor(Y_valid))
    test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
