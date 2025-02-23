import argparse
import yaml
import torch
import torch.nn as nn
import logging
from pathlib import Path

from src.load_data import get_dataset
from src.models.TimeSeriesTransformer import TimeSeriesTransformer
from src.models.TSTransformerEncoder import TSTransformerEncoder
from src.utils import get_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model(config, input_dim, num_classes):
    """Initialize model based on config."""
    if config['model']['type'] == 'patch_embed':
        model = TimeSeriesTransformer(
            input_timesteps=config['model']['input_timesteps'],
            in_channels=input_dim,
            patch_size=config['model']['patch_size'],
            embedding_dim=config['model']['embedding_dim'],
            pos_encoding=config['model']['encoding'],
            num_transformer_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            num_classes=num_classes
        )
    else:  # batch_norm
        model = TSTransformerEncoder(
            feat_dim=input_dim,
            max_len=config['model']['input_timesteps'],
            d_model=config['model']['embedding_dim'],
            n_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            pos_encoding=config['model']['encoding'],
            num_classes=num_classes
        )
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    accuracy = 100. * correct / total
    return total_loss / len(train_loader), accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    accuracy = 100. * correct / total
    return total_loss / len(val_loader), accuracy

def run_experiment(model, data, config):
    """Run training and evaluation."""
    device = torch.device(config['training']['device'])
    model = model.to(device)
    
    X_train, y_train, X_valid, y_valid, X_test, y_test, k_size, EPOCHS, t_names = data
    
    # Create dataloaders
    train_loader, valid_loader, test_loader = get_dataloaders(
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        batch_size=config['training']['batch_size']
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate']
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(config['training']['epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        logger.info(f'Epoch: {epoch}')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, epoch, val_acc, config)
    
    # Test best model
    model = load_best_checkpoint(model, config)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    return {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc
    }

def save_checkpoint(model, epoch, val_acc, config):
    """Save model checkpoint."""
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"model_{config['model']['type']}_{config['model']['encoding']}_epoch{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'config': config
    }, checkpoint_path)

def load_best_checkpoint(model, config):
    """Load the best checkpoint for testing."""
    checkpoint_dir = Path('checkpoints')
    checkpoints = list(checkpoint_dir.glob(f"model_{config['model']['type']}_{config['model']['encoding']}*.pt"))
    best_checkpoint = max(checkpoints, key=lambda x: torch.load(x)['val_acc'])
    
    checkpoint = torch.load(best_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset
    data = get_dataset(config['dataset']['name'])
    
    # Create model
    input_dim = data[0].shape[2]  # X_train.shape[2]
    num_classes = len(torch.unique(data[1]))  # unique values in y_train
    model = get_model(config, input_dim, num_classes)

    # Run experiment
    results = run_experiment(model, data, config)
    logger.info(f"Final Results: {results}")

if __name__ == '__main__':
    main()