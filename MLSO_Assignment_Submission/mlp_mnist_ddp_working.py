#!/usr/bin/env python3
"""
MLSO Assignment S2_24 PS4: Distributed Mini-Batch Neural Network Training
Multi-Layer Perceptron (MLP) for MNIST Digit Classification using PyTorch DDP

This implementation covers:
- MLP built from scratch using PyTorch
- Mini-batch training implementation
- Distributed training using PyTorch DDP (CPU backend)
- Per-epoch timing and communication overhead tracking
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for MNIST digit classification.
    Built from scratch using PyTorch.
    
    Architecture:
    - Input: 784 (28x28 flattened MNIST images)
    - Hidden layers: 512, 256, 128
    - Output: 10 (digit classes 0-9)
    - Activation: ReLU
    - Dropout: 0.2 for regularization
    """
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Flatten input: (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)


def setup_ddp(rank, world_size):
    """
    Setup Distributed Data Parallel (DDP) environment using CPU backend.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group with gloo backend (CPU)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Set device to CPU
    device = torch.device('cpu')
    
    return device


def cleanup_ddp():
    """Cleanup DDP process group."""
    dist.destroy_process_group()


def get_mnist_data(batch_size, world_size, rank):
    """
    Load and prepare MNIST dataset with distributed sampling.
    
    Args:
        batch_size: Batch size per process
        world_size: Total number of processes
        rank: Process rank
    
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    try:
        # Load datasets
        if rank == 0:
            print("Loading MNIST dataset...")
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        if rank == 0:
            print(f"Training samples: {len(train_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
        
        # Split test into validation and test
        val_size = 5000
        test_size = len(test_dataset) - val_size
        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
        
        # Create distributed samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=0,  # No multiprocessing for stability
            pin_memory=False  # Disabled for CPU
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            sampler=val_sampler,
            num_workers=0,  # No multiprocessing for stability
            pin_memory=False  # Disabled for CPU
        )
        
        return train_loader, val_loader, train_sampler
        
    except Exception as e:
        if rank == 0:
            print(f"Error loading MNIST dataset: {e}")
            print("Creating dummy data for testing...")
        
        # Create dummy data for testing
        dummy_train_data = torch.randn(1000, 1, 28, 28)
        dummy_train_labels = torch.randint(0, 10, (1000,))
        dummy_val_data = torch.randn(200, 1, 28, 28)
        dummy_val_labels = torch.randint(0, 10, (200,))
        
        # Create dummy datasets
        class DummyDataset:
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        train_dataset = DummyDataset(dummy_train_data, dummy_train_labels)
        val_dataset = DummyDataset(dummy_val_data, dummy_val_labels)
        
        # Create distributed samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            sampler=val_sampler,
            num_workers=0
        )
        
        return train_loader, val_loader, train_sampler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank):
    """
    Train for one epoch.
    
    Args:
        model: MLP model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        rank: Process rank
    
    Returns:
        train_loss: Average training loss
        train_acc: Training accuracy
        epoch_time: Time taken for the epoch
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    # Timing variables
    epoch_start = time.time()
    
    # Set sampler epoch for proper shuffling
    train_loader.sampler.set_epoch(epoch)
    
    # Progress bar (only for rank 0)
    if rank == 0:
        print(f"Training Epoch {epoch+1}...")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Synchronize gradients across processes (communication overhead)
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()
        
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Print progress every 100 batches (only rank 0)
        if rank == 0 and batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # Calculate epoch time
    epoch_time = time.time() - epoch_start
    
    # Synchronize metrics across processes
    train_loss_tensor = torch.tensor(train_loss).to(device)
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total).to(device)
    
    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    train_loss = train_loss_tensor.item() / dist.get_world_size()
    correct = correct_tensor.item()
    total = total_tensor.item()
    
    train_acc = 100. * correct / total
    
    return train_loss, train_acc, epoch_time


def validate(model, val_loader, criterion, device, rank):
    """
    Validate the model.
    
    Args:
        model: MLP model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        rank: Process rank
    
    Returns:
        val_loss: Validation loss
        val_acc: Validation accuracy
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    # Synchronize metrics across processes
    val_loss_tensor = torch.tensor(val_loss).to(device)
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total).to(device)
    
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    val_loss = val_loss_tensor.item() / dist.get_world_size()
    correct = correct_tensor.item()
    total = total_tensor.item()
    
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main_worker(rank, world_size, config):
    """
    Main worker function for distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Training configuration dictionary
    """
    print(f"Running DDP on rank {rank}.")
    
    # Setup DDP
    device = setup_ddp(rank, world_size)
    
    # Create model and move to device
    model = MLP(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=None)  # No device IDs for CPU
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['lr_scheduler_step'], 
        gamma=config['lr_scheduler_gamma']
    )
    
    # Get data loaders
    train_loader, val_loader, train_sampler = get_mnist_data(
        config['batch_size'], world_size, rank
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }
    
    print(f"Starting training on rank {rank}...")
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Train
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, rank
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, rank)
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        # Print progress (only rank 0)
        if rank == 0:
            print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Epoch Time: {epoch_time:.2f}s')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)
    
    # Save training history (only rank 0)
    if rank == 0:
        save_training_results(history, config, world_size)
    
    # Cleanup
    cleanup_ddp()


def save_training_results(history, config, world_size):
    """
    Save training results and generate plots.
    
    Args:
        history: Training history dictionary
        config: Training configuration
        world_size: Number of processes used
    """
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save configuration and results
    results = {
        'config': config,
        'world_size': world_size,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate plots
    create_training_plots(history, config, world_size)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING COMPLETED - SUMMARY STATISTICS")
    print("="*60)
    print(f"World Size: {world_size} processes")
    print(f"Total Epochs: {len(history['epoch_times'])}")
    print(f"Average Epoch Time: {np.mean(history['epoch_times']):.2f}s")
    print(f"Total Training Time: {np.sum(history['epoch_times']):.2f}s")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Best Val Accuracy: {max(history['val_acc']):.2f}%")
    print("="*60)


def create_training_plots(history, config, world_size):
    """
    Create and save training plots.
    
    Args:
        history: Training history dictionary
        config: Training configuration
        world_size: Number of processes used
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'MLP Training Results (World Size: {world_size})', fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Epoch time plot
    axes[1, 0].plot(epochs, history['epoch_times'], 'g-', marker='o')
    axes[1, 0].set_title('Epoch Training Time')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(True)
    
    # Learning curves comparison
    axes[1, 1].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[1, 1].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[1, 1].set_title('Learning Curves')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/training_plots_worldsize_{world_size}.png', dpi=300, bbox_inches='tight')
    print(f"Training plots saved as 'results/training_plots_worldsize_{world_size}.png'")


if __name__ == "__main__":
    # Training configuration
    config = {
        'input_size': 784,           # 28x28 MNIST images
        'hidden_sizes': [512, 256, 128],  # Hidden layer sizes
        'num_classes': 10,           # MNIST digits 0-9
        'dropout_rate': 0.2,         # Dropout for regularization
        'batch_size': 32,            # Reduced batch size for CPU
        'num_epochs': 5,             # Reduced epochs for faster testing
        'learning_rate': 0.001,      # Learning rate
        'weight_decay': 1e-5,        # L2 regularization
        'lr_scheduler_step': 3,      # LR scheduler step size
        'lr_scheduler_gamma': 0.5    # LR scheduler gamma
    }
    
    # Number of processes (world size)
    world_size = 2  # You can change this to test different numbers of processes
    
    print(f"Starting distributed training with {world_size} processes...")
    print(f"Configuration: {config}")
    print("Note: This version uses CPU backend (gloo) for distributed training.")
    print("If MNIST download fails, it will use dummy data for demonstration.")
    
    # Launch processes
    mp.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
