import torch

def get_dataloaders(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, batch_size=32):
    train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    valid_data = torch.utils.data.TensorDataset(torch.Tensor(X_valid), torch.Tensor(Y_valid))
    test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

class TUPEConfig:
    num_layers: int = 6
    num_heads: int = 8
    d_model: int = 128
    d_head: int = 0
    max_len: int = 256
    dropout: float = 0.1
    expansion_factor: int = 1
    relative_bias: bool = True
    bidirectional_bias: bool = True
    num_buckets: int = 32
    max_distance: int = 128

    def __post_init__(self):
        d_head, remainder = divmod(self.d_model, self.num_heads)
        assert remainder == 0, "`d_model` should be divisible by `num_heads`"
        self.d_head = d_head