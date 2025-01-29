import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enhanced data augmentation with MixUp
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# MixUp loss
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Data transforms
transform_train = transforms.Compose([
    transforms.Resize((144, 144)),  # Slightly larger for random crop
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MetricMonitor:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})
    
    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]
    
    def get_avg(self, metric_name):
        return self.metrics[metric_name]["avg"]

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    best_val_accuracy = 0.0
    patience = 12
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        metric_monitor = MetricMonitor()
        stream = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(stream):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss with mixup
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = (lam * predicted.eq(targets_a).float() + (1 - lam) * predicted.eq(targets_b).float()).sum()
            total = targets.size(0)
            accuracy = 100 * correct / total
            
            metric_monitor.update('loss', loss.item())
            metric_monitor.update('accuracy', accuracy.item())
            
            stream.set_postfix({
                'loss': f"{metric_monitor.get_avg('loss'):.4f}",
                'acc': f"{metric_monitor.get_avg('accuracy'):.2f}%"
            })
        
        # Save training metrics
        history['train_loss'].append(metric_monitor.get_avg('loss'))
        history['train_acc'].append(metric_monitor.get_avg('accuracy'))
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        metric_monitor_val = MetricMonitor()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum()
                total = targets.size(0)
                accuracy = 100 * correct / total
                
                metric_monitor_val.update('loss', loss.item())
                metric_monitor_val.update('accuracy', accuracy.item())
        
        val_loss = metric_monitor_val.get_avg('loss')
        val_accuracy = metric_monitor_val.get_avg('accuracy')
        
        # Save validation metrics
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Training Loss: {metric_monitor.get_avg("loss"):.4f}, Accuracy: {metric_monitor.get_avg("accuracy"):.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'history': history
            }, 'best_model_checkpoint.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    
    return best_val_accuracy, history

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    BASE_LR = 5e-5  # Reduced learning rate
    IMAGE_DIR = r'c:\sreejha_project\intedata'
    
    # Load and split datasets
    full_dataset = datasets.ImageFolder(os.path.join(IMAGE_DIR, 'train'), transform=transform_train)
    
    # Calculate splits
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = datasets.ImageFolder(
        os.path.join(IMAGE_DIR, 'test'),
        transform=transform_val_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = ImprovedCNN(num_classes=3).to(device)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with layer-wise learning rates
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': BASE_LR * 0.1},
        {'params': model.classifier.parameters(), 'lr': BASE_LR}
    ], weight_decay=0.02)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )
    
    # Train model
    best_accuracy, history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        NUM_EPOCHS, device
    )
    
    # Plot training history
    plot_training_history(history)
    
    print(f'Best validation accuracy: {best_accuracy:.2f}%')
    
    # Load best model and evaluate on test set
    checkpoint = torch.load('best_model_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            actual_labels.extend(targets.cpu().numpy())
    
    test_accuracy = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main()