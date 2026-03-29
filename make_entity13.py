import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from robustness.tools.breeds_helpers import ClassHierarchy, make_entity13, print_dataset_info, BreedsDatasetGenerator
from robustness import datasets, model_utils
from robustness.datasets import CustomImageNet
import numpy as np
import os
from tqdm import tqdm
import time
DEVICE = 'cuda:1'

# ==================== 1. Setup Entity13 ====================
info_dir = os.path.expanduser("~/imagenet_info")
data_dir = os.path.expanduser("~/imagenet")

hier = ClassHierarchy(info_dir)
print(f"# Levels in hierarchy: {np.max(list(hier.level_to_nodes.keys()))}")
print(f"# Nodes/level:", [f"Level {k}: {len(v)}" for k, v in hier.level_to_nodes.items()])

ret = make_entity13(info_dir, split="rand")
superclasses, subclass_split, label_map = ret
print(print_dataset_info(superclasses, subclass_split, label_map, hier.LEAF_NUM_TO_NAME))
print('Entity13 setup complete\n')


# ==================== 2. Entity13 Classifier ====================
class Entity13Classifier(nn.Module):
    def __init__(self, num_classes=13, pretrained=True, arch='resnet50'):
        super(Entity13Classifier, self).__init__()
        
        if arch == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            if pretrained:
                base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                base_model = resnet50(weights=None)
            self.feature_dim = 2048
        elif arch == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained:
                base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                base_model = resnet18(weights=None)
            self.feature_dim = 512
        else:
            raise ValueError(f"Architecture {arch} not supported")
        
        # Extract feature extractor
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        # New classification head for Entity13
        self.linear1 = nn.Linear(self.feature_dim, self.feature_dim)
        self.linear2 = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize new layers
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)
    
    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x):
        features = self.get_features(x)
        out = self.linear2(torch.relu(self.linear1(features)))
        return out
    
    def get_features_and_scores(self, x):
        features = self.get_features(x)
        scores = self.linear2(torch.relu(self.linear1(features)))
        return features, scores


# ==================== 3. Data Loading (CORRECTED) ====================
def get_entity13_loaders(data_dir, subclass_split, batch_size=128, num_workers=4):
    """
    Create train and test loaders for Entity13 dataset using CustomImageNet
    Following the official BREEDS documentation pattern
    """
    # Get train and test subclasses
    train_subclasses = subclass_split[0]
    test_subclasses = subclass_split[1]
    
    # Create datasets for source (train) and target (test) domains
    dataset_train = CustomImageNet(data_dir, train_subclasses)
    dataset_test = CustomImageNet(data_dir, test_subclasses)
    
    # Create loaders
    loaders_train = dataset_train.make_loaders(num_workers, batch_size)
    train_loader, _ = loaders_train  # We only use the train split
    
    loaders_test = dataset_test.make_loaders(num_workers, batch_size)
    _, test_loader = loaders_test  # We only use the val split for testing
    
    return train_loader, test_loader


# ==================== 4. Training Function ====================
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ==================== 5. Evaluation Function ====================
def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


# ==================== 6. Feature Extraction Function ====================
def extract_features_and_scores(model, dataloader, device='cuda'):
    """Extract features and scores from trained model"""
    model.eval()
    all_features = []
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Extracting features')
        for images, labels in pbar:
            images = images.to(device)
            
            # Get features and scores
            features, scores = model.get_features_and_scores(images)
            
            all_features.append(features.cpu())
            all_scores.append(scores.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_scores, all_labels


# ==================== 7. Main Training Loop ====================
def train_entity13_classifier(
    arch='resnet50',
    num_epochs=30,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=5e-4,
    save_path='entity13_classifier.pth'
):
    """
    Complete training pipeline for Entity13 classifier
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:1'
    print(f"Using device: {device}\n")
    
    # Create model
    print("Creating model...")
    model = Entity13Classifier(num_classes=13, pretrained=True, arch=arch)
    model = model.to(device)
    print(f"✓ Model created: {arch}")
    print(f"  - Feature dimension: {model.feature_dim}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Create dataloaders
    print("Loading data...")
    try:
        train_loader, test_loader = get_entity13_loaders(
            data_dir, subclass_split, batch_size, 4
        )
        print(f"✓ Data loaded")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}\n")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Data loading failed: {e}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                         momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones=[15, 25], 
                                               gamma=0.1)
    
    # Training loop
    print("Starting training...\n")
    best_acc = 0.0
    train_history = {'loss': [], 'acc': []}
    test_history = {'loss': [], 'acc': []}
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_history['loss'].append(test_loss)
        test_history['acc'].append(test_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"  ★ New best accuracy! Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'test_history': test_history,
            }, save_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}\n")
    
    return model, train_history, test_history


# ==================== 8. Run Training ====================
if __name__ == "__main__":
    # Train the model
    model, train_history, test_history = train_entity13_classifier(
        arch='resnet50',
        num_epochs=30,
        batch_size=64,
        learning_rate=0.001,
        weight_decay=5e-4,
        save_path='entity13_resnet50_classifier.pth'
    )
    
    # Extract features for KOPE pipeline
    print("\n" + "="*60)
    print("Extracting features for KOPE pipeline...")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best model
    checkpoint = torch.load('entity13_resnet50_classifier.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Get dataloaders again
    train_loader, test_loader = get_entity13_loaders(
        data_dir, subclass_split, batch_size=128, num_workers=4
    )
    
    # Extract train features
    print("Extracting training features...")
    train_features, train_scores, train_labels = extract_features_and_scores(
        model, train_loader, device
    )
    print(f"✓ Train features extracted: {train_features.shape}")
    
    # Extract test features
    print("\nExtracting test features...")
    test_features, test_scores, test_labels = extract_features_and_scores(
        model, test_loader, device
    )
    print(f"✓ Test features extracted: {test_features.shape}")
    
    # Save features
    print("\nSaving features...")
    torch.save({
        'train_features': train_features,
        'train_scores': train_scores,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_scores': test_scores,
        'test_labels': test_labels,
    }, 'entity13_features.pth')
    
    print(f"✓ Features saved to: entity13_features.pth")
    print("\n" + "="*60)
    print("Ready for KOPE decoy generation!")
    print("="*60)