# PyTorch Model Development

*Duration: 2-3 weeks*

## nn.Module System

The `nn.Module` is the fundamental building block of PyTorch neural networks. Understanding it deeply is crucial for effective model development.

### Creating Custom Modules

#### Basic Custom Module Structure
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    """
    A basic neural network demonstrating nn.Module fundamentals
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicModel, self).__init__()
        
        # Define layers as module attributes
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # First layer with ReLU activation
        x = F.relu(self.linear1(x))
        x = self.batch_norm(x)
        x = self.dropout(x)
        
        # Second layer with ReLU activation
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        
        # Output layer (no activation for regression)
        x = self.linear3(x)
        
        return x

# Example usage
model = BasicModel(input_size=784, hidden_size=256, output_size=10)
print(f"Model structure:\n{model}")

# Check model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

#### Advanced Custom Module with Submodules
```python
class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2d + BatchNorm + ReLU + optional Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = None
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections
    """
    def __init__(self, channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvBlock(channels, channels, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(channels)
        
        # Skip connection adjustment for dimension matching
        if stride != 1:
            self.skip_connection = nn.Conv2d(channels, channels, 1, stride, bias=False)
        else:
            self.skip_connection = None
    
    def forward(self, x):
        residual = x
        
        # Main path
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        
        # Skip connection
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
        
        # Add residual and apply ReLU
        out += residual
        out = F.relu(out)
        
        return out

class CustomCNN(nn.Module):
    """
    Custom CNN using modular design
    """
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # Input processing
        self.input_conv = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        # Feature extraction
        self.conv_layers = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2)
        )
        
        # Global average pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input processing
        x = self.input_conv(x)
        x = self.max_pool(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Feature extraction
        x = self.conv_layers(x)
        
        # Global pooling and classification
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# Example usage
model = CustomCNN(num_classes=1000)
sample_input = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
output = model(sample_input)
print(f"Output shape: {output.shape}")
```

### Layer Composition and ModuleList/ModuleDict

#### Using nn.ModuleList for Dynamic Layers
```python
class DynamicMLP(nn.Module):
    """
    Multi-layer perceptron with configurable number of layers
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        super(DynamicMLP, self).__init__()
        
        # Build layer dimensions
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Create layers using ModuleList
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        # Activation function
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation
        
        # Dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2) for _ in range(len(hidden_sizes))
        ])
    
    def forward(self, x):
        # Process through all but the last layer
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropouts[i](x)
        
        # Last layer (no activation/dropout)
        x = self.layers[-1](x)
        return x

# Example usage
model = DynamicMLP(
    input_size=784,
    hidden_sizes=[512, 256, 128],
    output_size=10,
    activation='relu'
)
```

#### Using nn.ModuleDict for Named Components
```python
class MultiTaskModel(nn.Module):
    """
    Model with multiple task-specific heads
    """
    def __init__(self, backbone_size=512):
        super(MultiTaskModel, self).__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, backbone_size),
            nn.ReLU()
        )
        
        # Task-specific heads using ModuleDict
        self.task_heads = nn.ModuleDict({
            'classification': nn.Sequential(
                nn.Linear(backbone_size, 256),
                nn.ReLU(),
                nn.Linear(256, 10)  # 10 classes
            ),
            'regression': nn.Sequential(
                nn.Linear(backbone_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)   # Single value
            ),
            'segmentation': nn.Sequential(
                nn.Linear(backbone_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 784)  # Pixel-wise output
            )
        })
    
    def forward(self, x, task='classification'):
        # Shared feature extraction
        features = self.backbone(x)
        
        # Task-specific processing
        if task in self.task_heads:
            return self.task_heads[task](features)
        else:
            # Return all task outputs
            return {task_name: head(features) for task_name, head in self.task_heads.items()}

# Example usage
model = MultiTaskModel()
sample_input = torch.randn(32, 784)

# Single task inference
classification_output = model(sample_input, task='classification')
print(f"Classification output shape: {classification_output.shape}")

# Multi-task inference
all_outputs = model(sample_input, task='all')
for task_name, output in all_outputs.items():
    print(f"{task_name} output shape: {output.shape}")
```

### Parameter Management

#### Understanding and Managing Parameters
```python
class ParameterExplorer(nn.Module):
    """
    Demonstrate parameter management techniques
    """
    def __init__(self):
        super(ParameterExplorer, self).__init__()
        
        # Standard parameters (learnable)
        self.linear = nn.Linear(10, 5)
        
        # Custom parameters
        self.custom_weight = nn.Parameter(torch.randn(5, 3))
        self.custom_bias = nn.Parameter(torch.zeros(3))
        
        # Non-learnable parameters (buffers)
        self.register_buffer('running_mean', torch.zeros(5))
        self.register_buffer('running_var', torch.ones(5))
        
        # Frozen parameters
        self.frozen_layer = nn.Linear(5, 3)
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
    
    def forward(self, x):
        x = self.linear(x)
        
        # Use buffers for normalization (example)
        x = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        
        # Custom parameter computation
        x = torch.matmul(x, self.custom_weight) + self.custom_bias
        
        # Frozen layer
        x = self.frozen_layer(x)
        
        return x
    
    def update_buffers(self, new_mean, new_var):
        """Update non-learnable buffers"""
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)

# Parameter analysis
model = ParameterExplorer()

print("=== Parameter Analysis ===")
print(f"Named parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

print(f"\nNamed buffers:")
for name, buffer in model.named_buffers():
    print(f"  {name}: {buffer.shape}")

print(f"\nParameter groups:")
# Separate learnable and frozen parameters
learnable_params = [p for p in model.parameters() if p.requires_grad]
frozen_params = [p for p in model.parameters() if not p.requires_grad]

print(f"  Learnable parameters: {sum(p.numel() for p in learnable_params):,}")
print(f"  Frozen parameters: {sum(p.numel() for p in frozen_params):,}")
```

### Forward and Backward Passes with Hooks

#### Understanding Forward and Backward Hooks
```python
class HookDemonstration(nn.Module):
    """
    Demonstrate forward and backward hooks for debugging and analysis
    """
    def __init__(self):
        super(HookDemonstration, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, 10)
        
        # Storage for hook data
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for each layer
        self.conv1.register_forward_hook(forward_hook('conv1'))
        self.conv2.register_forward_hook(forward_hook('conv2'))
        self.conv3.register_forward_hook(forward_hook('conv3'))
        
        self.conv1.register_backward_hook(backward_hook('conv1'))
        self.conv2.register_backward_hook(backward_hook('conv2'))
        self.conv3.register_backward_hook(backward_hook('conv3'))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_activation_stats(self):
        """Get statistics about activations"""
        stats = {}
        for name, activation in self.activations.items():
            stats[name] = {
                'shape': activation.shape,
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'min': activation.min().item(),
                'max': activation.max().item(),
                'sparsity': (activation == 0).float().mean().item()
            }
        return stats
    
    def get_gradient_stats(self):
        """Get statistics about gradients"""
        stats = {}
        for name, gradient in self.gradients.items():
            if gradient is not None:
                stats[name] = {
                    'shape': gradient.shape,
                    'mean': gradient.mean().item(),
                    'std': gradient.std().item(),
                    'norm': gradient.norm().item()
                }
        return stats

# Example usage with hooks
model = HookDemonstration()
sample_input = torch.randn(2, 3, 32, 32)
sample_target = torch.randint(0, 10, (2,))

# Forward pass
output = model(sample_input)

# Backward pass
loss = F.cross_entropy(output, sample_target)
loss.backward()

# Analyze activations and gradients
print("=== Activation Statistics ===")
for layer, stats in model.get_activation_stats().items():
    print(f"{layer}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, sparsity={stats['sparsity']:.4f}")

print("\n=== Gradient Statistics ===")
for layer, stats in model.get_gradient_stats().items():
    print(f"{layer}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, norm={stats['norm']:.4f}")
```

## Training Workflow

### DataLoader and Datasets

#### Creating Custom Datasets
```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    """
    Custom dataset for image classification
    """
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to csv file with image paths and labels
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on images
            target_transform (callable, optional): Optional transform to be applied on labels
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_path'])
        image = Image.open(img_name).convert('RGB')
        
        # Load label
        label = self.data_frame.iloc[idx]['label']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets"""
        labels = self.data_frame['label'].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)

class CustomTextDataset(Dataset):
    """
    Custom dataset for text classification
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Example data transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = CustomImageDataset('train.csv', 'train_images/', transform=train_transforms)
val_dataset = CustomImageDataset('val.csv', 'val_images/', transform=val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # Faster GPU transfer
    drop_last=True    # Drop incomplete batches
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

#### Advanced DataLoader Techniques
```python
from torch.utils.data import WeightedRandomSampler

# Handling imbalanced datasets with weighted sampling
def create_weighted_sampler(dataset):
    """Create weighted sampler for imbalanced datasets"""
    # Get class weights
    class_weights = dataset.get_class_weights()
    
    # Get sample weights
    labels = [dataset[i][1] for i in range(len(dataset))]
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

# Custom collate function for variable-length sequences
def custom_collate_fn(batch):
    """Custom collate function for handling variable-length data"""
    images, labels = zip(*batch)
    
    # Stack images (assuming they're the same size after transforms)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    
    return images, labels

# Advanced DataLoader with custom sampler
weighted_sampler = create_weighted_sampler(train_dataset)
train_loader_balanced = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=weighted_sampler,  # Use weighted sampler instead of shuffle
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate_fn
)
```

### Loss Functions

#### Common Loss Functions and Their Use Cases
```python
import torch.nn as nn
import torch.nn.functional as F

class LossFunctionExamples:
    """Comprehensive examples of PyTorch loss functions"""
    
    @staticmethod
    def classification_losses():
        """Examples of classification loss functions"""
        # Sample data
        batch_size, num_classes = 4, 10
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # 1. Cross Entropy Loss (most common for classification)
        ce_loss = nn.CrossEntropyLoss()
        loss_ce = ce_loss(logits, targets)
        print(f"CrossEntropy Loss: {loss_ce.item():.4f}")
        
        # 2. Cross Entropy with class weights (for imbalanced datasets)
        class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0, 3.0, 1.0, 1.0, 2.5, 1.0, 1.0])
        ce_weighted = nn.CrossEntropyLoss(weight=class_weights)
        loss_ce_weighted = ce_weighted(logits, targets)
        print(f"Weighted CrossEntropy Loss: {loss_ce_weighted.item():.4f}")
        
        # 3. Negative Log Likelihood (when you have log probabilities)
        log_probs = F.log_softmax(logits, dim=1)
        nll_loss = nn.NLLLoss()
        loss_nll = nll_loss(log_probs, targets)
        print(f"NLL Loss: {loss_nll.item():.4f}")
        
        # 4. Binary Cross Entropy (for binary classification)
        binary_logits = torch.randn(batch_size, 1)
        binary_targets = torch.randint(0, 2, (batch_size, 1)).float()
        bce_loss = nn.BCEWithLogitsLoss()
        loss_bce = bce_loss(binary_logits, binary_targets)
        print(f"BCE Loss: {loss_bce.item():.4f}")
        
        return {
            'cross_entropy': loss_ce,
            'weighted_cross_entropy': loss_ce_weighted,
            'nll': loss_nll,
            'bce': loss_bce
        }
    
    @staticmethod
    def regression_losses():
        """Examples of regression loss functions"""
        # Sample data
        predictions = torch.randn(10, 1)
        targets = torch.randn(10, 1)
        
        # 1. Mean Squared Error (L2 loss)
        mse_loss = nn.MSELoss()
        loss_mse = mse_loss(predictions, targets)
        print(f"MSE Loss: {loss_mse.item():.4f}")
        
        # 2. Mean Absolute Error (L1 loss)
        mae_loss = nn.L1Loss()
        loss_mae = mae_loss(predictions, targets)
        print(f"MAE Loss: {loss_mae.item():.4f}")
        
        # 3. Smooth L1 Loss (Huber loss)
        smooth_l1_loss = nn.SmoothL1Loss()
        loss_smooth_l1 = smooth_l1_loss(predictions, targets)
        print(f"Smooth L1 Loss: {loss_smooth_l1.item():.4f}")
        
        return {
            'mse': loss_mse,
            'mae': loss_mae,
            'smooth_l1': loss_smooth_l1
        }

# Custom loss functions
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice_coeff

# Example usage of custom losses
focal_loss = FocalLoss(alpha=1, gamma=2)
dice_loss = DiceLoss(smooth=1.0)
```

### Optimizers

#### Comprehensive Optimizer Examples
```python
import torch.optim as optim

class OptimizerExamples:
    """Comprehensive examples of PyTorch optimizers"""
    
    def __init__(self, model):
        self.model = model
    
    def basic_optimizers(self):
        """Basic optimizer configurations"""
        optimizers = {}
        
        # 1. Stochastic Gradient Descent (SGD)
        optimizers['sgd'] = optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
        
        # 2. Adam (most popular)
        optimizers['adam'] = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
            amsgrad=False
        )
        
        # 3. AdamW (Adam with decoupled weight decay)
        optimizers['adamw'] = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # 4. RMSprop
        optimizers['rmsprop'] = optim.RMSprop(
            self.model.parameters(),
            lr=0.01,
            alpha=0.99,
            eps=1e-8,
            weight_decay=1e-4,
            momentum=0.9
        )
        
        # 5. AdaGrad
        optimizers['adagrad'] = optim.Adagrad(
            self.model.parameters(),
            lr=0.01,
            lr_decay=0,
            weight_decay=1e-4,
            eps=1e-10
        )
        
        return optimizers
    
    def parameter_groups(self):
        """Different learning rates for different parts of the model"""
        # Different learning rates for different layers
        param_groups = [
            {
                'params': self.model.backbone.parameters(),
                'lr': 0.0001,  # Lower learning rate for pretrained backbone
                'weight_decay': 1e-4
            },
            {
                'params': self.model.classifier.parameters(),
                'lr': 0.001,   # Higher learning rate for new classifier
                'weight_decay': 1e-3
            }
        ]
        
        optimizer = optim.Adam(param_groups)
        return optimizer
    
    def layer_wise_lr_decay(self, base_lr=0.001, decay_rate=0.9):
        """Layer-wise learning rate decay"""
        param_groups = []
        
        # Get all layer names and parameters
        layer_names = []
        for name, param in self.model.named_parameters():
            layer_name = name.split('.')[0]  # Get first part of parameter name
            if layer_name not in layer_names:
                layer_names.append(layer_name)
        
        # Create parameter groups with decaying learning rates
        for i, layer_name in enumerate(layer_names):
            layer_params = []
            for name, param in self.model.named_parameters():
                if name.startswith(layer_name):
                    layer_params.append(param)
            
            lr = base_lr * (decay_rate ** i)
            param_groups.append({
                'params': layer_params,
                'lr': lr,
                'name': layer_name
            })
        
        optimizer = optim.Adam(param_groups)
        return optimizer

# Learning rate schedulers
class SchedulerExamples:
    """Examples of learning rate schedulers"""
    
    @staticmethod
    def create_schedulers(optimizer):
        schedulers = {}
        
        # 1. Step LR (decay by factor every few epochs)
        schedulers['step'] = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
        
        # 2. Multi-step LR (decay at specific epochs)
        schedulers['multistep'] = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
        
        # 3. Exponential LR (exponential decay)
        schedulers['exponential'] = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
        
        # 4. Cosine Annealing
        schedulers['cosine'] = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  # Maximum number of iterations
            eta_min=1e-6  # Minimum learning rate
        )
        
        # 5. Reduce on Plateau (adaptive based on metrics)
        schedulers['plateau'] = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 'min' for loss, 'max' for accuracy
            factor=0.5,
            patience=10,
            threshold=0.0001,
            min_lr=1e-6
        )
        
        # 6. Cosine Annealing with Warm Restarts
        schedulers['cosine_restart'] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Number of iterations for the first restart
            T_mult=2  # A factor increases T_i after a restart
        )
        
        # 7. One Cycle LR (popular for fast training)
        schedulers['onecycle'] = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            total_steps=1000,  # Total number of steps
            pct_start=0.3,  # Percentage of cycle spent increasing lr
            anneal_strategy='cos'
        )
        
        return schedulers

# Example usage
model = CustomCNN(num_classes=10)
optimizer_examples = OptimizerExamples(model)

# Get different optimizers
optimizers = optimizer_examples.basic_optimizers()

# Parameter groups example
param_group_optimizer = optimizer_examples.parameter_groups()

# Layer-wise LR decay
layer_wise_optimizer = optimizer_examples.layer_wise_lr_decay()

# Schedulers
scheduler_examples = SchedulerExamples()
schedulers = scheduler_examples.create_schedulers(optimizers['adam'])
```

### Training Loops

#### Complete Training Loop with Best Practices
```python
import time
from collections import defaultdict
import matplotlib.pyplot as plt

class Trainer:
    """
    Comprehensive training class with best practices
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler=None, device='cuda', save_dir='./checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        # Training history
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Progress tracking
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional, good for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == targets.data)
            total_samples += batch_size
            
            # Print progress
            if batch_idx % 100 == 0:
                current_loss = running_loss / total_samples
                current_acc = running_corrects.double() / total_samples
                elapsed = time.time() - start_time
                
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                      f'Loss: {current_loss:.4f}, Acc: {current_acc:.4f}, '
                      f'Time: {elapsed:.2f}s')
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * batch_size
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == targets.data)
                total_samples += batch_size
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': dict(self.history),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }
        
        # Save current checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = defaultdict(list, checkpoint['history'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        
        return checkpoint['epoch']
    
    def train(self, num_epochs, resume_from=None):
        """Complete training loop"""
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming training from epoch {start_epoch}")
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training on {len(self.train_loader.dataset)} samples")
        print(f"Validating on {len(self.val_loader.dataset)} samples")
        
        for epoch in range(start_epoch, num_epochs):
            print(f'\nEpoch {epoch}/{num_epochs-1}')
            print('-' * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint and check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
            
            # Save checkpoint every 10 epochs and if best
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print('\nTraining completed!')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss vs Epochs')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy vs Epochs')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['lr'])
        axes[1, 0].set_title('Learning Rate vs Epochs')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Combined loss/accuracy plot
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(self.history['val_loss'], 'b-', label='Val Loss')
        line2 = ax2.plot(self.history['val_acc'], 'r-', label='Val Acc')
        
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='b')
        ax2.set_ylabel('Accuracy', color='r')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.show()

# Example usage
model = CustomCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Train the model
history = trainer.train(num_epochs=100)
```

### Validation and Early Stopping

#### Early Stopping Implementation
```python
class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    """
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore model weights from best epoch
            verbose (bool): Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def __call__(self, val_loss, model):
        """
        Check if training should stop
        
        Args:
            val_loss (float): Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
            
            return False
        else:
            # No improvement
            self.wait += 1
            
            if self.verbose:
                print(f"No improvement for {self.wait} epochs (best: {self.best_loss:.6f})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("Restored best weights")
                
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement")
                
                return True
            
            return False

# Enhanced trainer with early stopping
class TrainerWithEarlyStopping(Trainer):
    """Extended trainer with early stopping capability"""
    
    def __init__(self, *args, early_stopping_patience=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=True
        )
    
    def train(self, num_epochs, resume_from=None):
        """Training loop with early stopping"""
        start_epoch = 0
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming training from epoch {start_epoch}")
        
        print(f"Starting training with early stopping (patience={self.early_stopping.patience})")
        
        for epoch in range(start_epoch, num_epochs):
            print(f'\nEpoch {epoch}/{num_epochs-1}')
            print('-' * 50)
            
            # Training and validation
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Check for early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
            
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print('\nTraining completed!')
        self.plot_training_history()
        return self.history

# K-Fold Cross Validation
class KFoldValidator:
    """
    K-Fold cross validation for robust model evaluation
    """
    def __init__(self, model_class, model_kwargs, k_folds=5):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.k_folds = k_folds
    
    def validate(self, dataset, criterion, optimizer_class, optimizer_kwargs, 
                 num_epochs=50, batch_size=32):
        """
        Perform k-fold cross validation
        """
        from sklearn.model_selection import KFold
        import numpy as np
        
        # Prepare data indices
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
            print(f"\nFold {fold + 1}/{self.k_folds}")
            print("=" * 50)
            
            # Create data subsets
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            
            # Create fresh model for this fold
            model = self.model_class(**self.model_kwargs)
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            
            # Train model
            trainer = TrainerWithEarlyStopping(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                save_dir=f'./fold_{fold}_checkpoints'
            )
            
            history = trainer.train(num_epochs)
            
            # Record fold results
            fold_results.append({
                'fold': fold,
                'best_val_loss': min(history['val_loss']),
                'best_val_acc': max(history['val_acc']),
                'final_val_loss': history['val_loss'][-1],
                'final_val_acc': history['val_acc'][-1],
                'history': history
            })
        
        # Aggregate results
        avg_val_loss = np.mean([result['best_val_loss'] for result in fold_results])
        avg_val_acc = np.mean([result['best_val_acc'] for result in fold_results])
        std_val_loss = np.std([result['best_val_loss'] for result in fold_results])
        std_val_acc = np.std([result['best_val_acc'] for result in fold_results])
        
        print(f"\n{self.k_folds}-Fold Cross Validation Results:")
        print(f"Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        print(f"Average Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
        
        return fold_results, {
            'avg_val_loss': avg_val_loss,
            'avg_val_acc': avg_val_acc,
            'std_val_loss': std_val_loss,
            'std_val_acc': std_val_acc
        }

# Example usage
validator = KFoldValidator(
    model_class=CustomCNN,
    model_kwargs={'num_classes': 10},
    k_folds=5
)

# Run cross-validation
fold_results, summary = validator.validate(
    dataset=train_dataset,
    criterion=nn.CrossEntropyLoss(),
    optimizer_class=optim.Adam,
    optimizer_kwargs={'lr': 0.001, 'weight_decay': 1e-4},
    num_epochs=50,
    batch_size=32
)
```

## PyTorch Ecosystem

The PyTorch ecosystem provides specialized libraries for different domains, making it easier to implement domain-specific solutions.

### torchvision for Computer Vision

#### Core Computer Vision Components
```python
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image

# 1. Pre-trained Models
class PretrainedModelExamples:
    """Examples of using pre-trained models from torchvision"""
    
    @staticmethod
    def load_pretrained_models():
        """Load various pre-trained models"""
        models_dict = {}
        
        # ResNet family
        models_dict['resnet18'] = models.resnet18(pretrained=True)
        models_dict['resnet50'] = models.resnet50(pretrained=True)
        models_dict['resnet101'] = models.resnet101(pretrained=True)
        
        # VGG family
        models_dict['vgg16'] = models.vgg16(pretrained=True)
        models_dict['vgg19'] = models.vgg19(pretrained=True)
        
        # DenseNet family
        models_dict['densenet121'] = models.densenet121(pretrained=True)
        models_dict['densenet169'] = models.densenet169(pretrained=True)
        
        # EfficientNet family
        models_dict['efficientnet_b0'] = models.efficientnet_b0(pretrained=True)
        models_dict['efficientnet_b7'] = models.efficientnet_b7(pretrained=True)
        
        # Vision Transformer
        models_dict['vit_b_16'] = models.vit_b_16(pretrained=True)
        
        # Modern architectures
        models_dict['mobilenet_v3_large'] = models.mobilenet_v3_large(pretrained=True)
        models_dict['regnet_y_400mf'] = models.regnet_y_400mf(pretrained=True)
        
        return models_dict
    
    @staticmethod
    def fine_tune_model(model_name, num_classes, freeze_backbone=True):
        """Fine-tune a pre-trained model for custom dataset"""
        # Load pre-trained model
        if model_name.startswith('resnet'):
            model = getattr(models, model_name)(pretrained=True)
            # Replace final layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            
            if freeze_backbone:
                # Freeze all layers except the final one
                for param in model.parameters():
                    param.requires_grad = False
                model.fc.weight.requires_grad = True
                model.fc.bias.requires_grad = True
        
        elif model_name.startswith('vgg'):
            model = getattr(models, model_name)(pretrained=True)
            # Replace classifier
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, num_classes)
            
            if freeze_backbone:
                for param in model.features.parameters():
                    param.requires_grad = False
        
        elif model_name.startswith('densenet'):
            model = getattr(models, model_name)(pretrained=True)
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, num_classes)
            
            if freeze_backbone:
                for param in model.features.parameters():
                    param.requires_grad = False
        
        return model

# 2. Advanced Transforms
class AdvancedTransforms:
    """Advanced image transformation techniques"""
    
    @staticmethod
    def create_transform_pipeline(mode='train', image_size=224):
        """Create comprehensive transform pipeline"""
        
        if mode == 'train':
            transform = transforms.Compose([
                # Resize and crop
                transforms.Resize(int(image_size * 1.14)),
                transforms.RandomCrop(image_size),
                
                # Geometric transforms
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
                                      scale=(0.9, 1.1), shear=5),
                
                # Color transforms
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                
                # Advanced augmentations
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.3),
                
                # Normalization
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
                
                # Optional: Random erasing
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), 
                                       ratio=(0.3, 3.3))
            ])
        
        else:  # validation/test
            transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        return transform

# Example usage
pretrained_examples = PretrainedModelExamples()
models_dict = pretrained_examples.load_pretrained_models()

# Fine-tune ResNet18 for 100 classes
fine_tuned_model = pretrained_examples.fine_tune_model('resnet18', num_classes=100)

# Create advanced transforms
transform_utils = AdvancedTransforms()
train_transform = transform_utils.create_transform_pipeline('train')
val_transform = transform_utils.create_transform_pipeline('val')
```

### torchaudio for Audio Processing

#### Audio Processing Pipeline
```python
import torchaudio
import torchaudio.transforms as audio_transforms
import torchaudio.functional as audio_F

class AudioProcessingPipeline:
    """Comprehensive audio processing using torchaudio"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def load_and_preprocess_audio(self, file_path):
        """Load and preprocess audio file"""
        # Load audio
        waveform, original_sr = torchaudio.load(file_path)
        
        # Resample if necessary
        if original_sr != self.sample_rate:
            resampler = audio_transforms.Resample(original_sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def create_audio_transforms(self):
        """Create audio transformation pipeline"""
        transforms_dict = {}
        
        # Spectral transforms
        transforms_dict['mel_spectrogram'] = audio_transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0.0,
            f_max=self.sample_rate / 2
        )
        
        transforms_dict['mfcc'] = audio_transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_fft': 1024,
                'hop_length': 512,
                'n_mels': 128
            }
        )
        
        return transforms_dict

class AudioClassificationModel(nn.Module):
    """CNN model for audio classification using spectrograms"""
    
    def __init__(self, num_classes, input_channels=1):
        super(AudioClassificationModel, self).__init__()
        
        # Convolutional layers for spectrogram processing
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.25)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# Example usage
audio_processor = AudioProcessingPipeline(sample_rate=16000)
audio_model = AudioClassificationModel(num_classes=10)
```

### torchtext for NLP

#### Modern Text Processing
```python
# Modern text processing using transformers
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from collections import Counter
import re

class TextClassificationModel(nn.Module):
    """BERT-based text classification model"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super(TextClassificationModel, self).__init__()
        
        # Load pre-trained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Freeze BERT layers (optional)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

class CustomTextDataset(Dataset):
    """Custom dataset for text classification with BERT tokenizer"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Example usage
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = TextClassificationModel('bert-base-uncased', num_classes=3)
```

### Domain-specific Libraries

#### Computer Vision Extensions
```python
# Object Detection with torchvision
import torchvision.models.detection as detection_models

# Load pre-trained object detection models
faster_rcnn = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
mask_rcnn = detection_models.maskrcnn_resnet50_fpn(pretrained=True)
ssd = detection_models.ssd300_vgg16(pretrained=True)

# Segmentation models
deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=True)
fcn = models.segmentation.fcn_resnet101(pretrained=True)

# Video models
video_resnet = models.video.r3d_18(pretrained=True)
video_mobilenet = models.video.mc3_18(pretrained=True)
```

#### Scientific Computing Extensions
```python
# PyTorch Geometric for Graph Neural Networks
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    
    class GraphNeuralNetwork(nn.Module):
        def __init__(self, num_features, hidden_dim, num_classes):
            super(GraphNeuralNetwork, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.classifier(x)
            return x
    
except ImportError:
    print("PyTorch Geometric not installed")

# PyTorch Lightning for structured training
try:
    import pytorch_lightning as pl
    
    class LightningModel(pl.LightningModule):
        def __init__(self, model, learning_rate=1e-3):
            super().__init__()
            self.model = model
            self.learning_rate = learning_rate
            self.criterion = nn.CrossEntropyLoss()
        
        def forward(self, x):
            return self.model(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log('train_loss', loss)
            return loss
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log('val_loss', loss)
            return loss
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
except ImportError:
    print("PyTorch Lightning not installed")
```

## Advanced PyTorch Features

### Hooks and Module Inspection

#### Forward and Backward Hooks for Debugging
```python
class ModelInspector:
    """Tool for inspecting model internals using hooks"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.hooks = []
    
    def register_hooks(self, layer_names=None):
        """Register hooks for specified layers or all layers"""
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for all named modules
        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                # Register forward hook
                handle_fwd = module.register_forward_hook(get_activation(name))
                # Register backward hook
                handle_bwd = module.register_backward_hook(get_gradient(name))
                
                self.hooks.extend([handle_fwd, handle_bwd])
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_layer_statistics(self):
        """Get statistics about activations and gradients"""
        stats = {}
        
        # Activation statistics
        for name, activation in self.activations.items():
            if activation.numel() > 0:
                stats[f"{name}_activation"] = {
                    'shape': list(activation.shape),
                    'mean': activation.mean().item(),
                    'std': activation.std().item(),
                    'min': activation.min().item(),
                    'max': activation.max().item(),
                    'sparsity': (activation == 0).float().mean().item()
                }
        
        # Gradient statistics
        for name, gradient in self.gradients.items():
            if gradient.numel() > 0:
                stats[f"{name}_gradient"] = {
                    'shape': list(gradient.shape),
                    'mean': gradient.mean().item(),
                    'std': gradient.std().item(),
                    'norm': gradient.norm().item()
                }
        
        return stats
    
    def visualize_activations(self, layer_name, save_path=None):
        """Visualize activations for a specific layer"""
        import matplotlib.pyplot as plt
        
        if layer_name not in self.activations:
            print(f"No activations found for layer: {layer_name}")
            return
        
        activation = self.activations[layer_name]
        
        # Handle different tensor shapes
        if len(activation.shape) == 4:  # Conv layer (B, C, H, W)
            # Visualize first few channels of first sample
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for i in range(min(8, activation.shape[1])):
                ax = axes[i // 4, i % 4]
                ax.imshow(activation[0, i].cpu().numpy(), cmap='viridis')
                ax.set_title(f'Channel {i}')
                ax.axis('off')
        
        elif len(activation.shape) == 2:  # Dense layer (B, F)
            # Histogram of activation values
            plt.figure(figsize=(10, 6))
            plt.hist(activation[0].cpu().numpy(), bins=50, alpha=0.7)
            plt.title(f'Activation Distribution - {layer_name}')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Example usage
class ExampleCNN(nn.Module):
    def __init__(self):
        super(ExampleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Use the inspector
model = ExampleCNN()
inspector = ModelInspector(model)
inspector.register_hooks(['conv1', 'conv2', 'fc1'])

# Forward and backward pass
sample_input = torch.randn(1, 3, 32, 32)
sample_target = torch.randint(0, 10, (1,))

output = model(sample_input)
loss = F.cross_entropy(output, sample_target)
loss.backward()

# Get statistics
stats = inspector.get_layer_statistics()
for layer, stat in stats.items():
    print(f"{layer}: {stat}")

# Clean up
inspector.remove_hooks()
```

#### Custom Hooks for Gradient Clipping and Monitoring
```python
class GradientMonitor:
    """Monitor and clip gradients during training"""
    
    def __init__(self, model, clip_value=1.0):
        self.model = model
        self.clip_value = clip_value
        self.gradient_norms = []
        self.hooks = []
        
        # Register parameter hooks
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._gradient_hook(name))
                self.hooks.append(hook)
    
    def _gradient_hook(self, param_name):
        def hook(grad):
            # Record gradient norm
            grad_norm = grad.norm().item()
            self.gradient_norms.append((param_name, grad_norm))
            
            # Clip gradient if necessary
            if grad_norm > self.clip_value:
                grad.data = grad.data * (self.clip_value / grad_norm)
                print(f"Clipped gradient for {param_name}: {grad_norm:.4f} -> {self.clip_value}")
            
            return grad
        return hook
    
    def get_gradient_summary(self):
        """Get summary of gradient norms"""
        if not self.gradient_norms:
            return {}
        
        # Group by parameter name
        from collections import defaultdict
        param_norms = defaultdict(list)
        
        for param_name, norm in self.gradient_norms:
            param_norms[param_name].append(norm)
        
        summary = {}
        for param_name, norms in param_norms.items():
            summary[param_name] = {
                'mean': sum(norms) / len(norms),
                'max': max(norms),
                'min': min(norms),
                'count': len(norms)
            }
        
        return summary
    
    def clear_history(self):
        """Clear gradient history"""
        self.gradient_norms = []
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Feature Visualization Hooks
class FeatureVisualizer:
    """Visualize learned features in CNN layers"""
    
    def __init__(self, model):
        self.model = model
        self.feature_maps = {}
    
    def register_feature_hooks(self, layer_names):
        """Register hooks to capture feature maps"""
        def get_features(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                module.register_forward_hook(get_features(name))
    
    def visualize_filters(self, layer_name, num_filters=16):
        """Visualize learned filters"""
        import matplotlib.pyplot as plt
        
        # Get the layer
        layer = dict(self.model.named_modules())[layer_name]
        
        if hasattr(layer, 'weight'):
            weights = layer.weight.data
            
            if len(weights.shape) == 4:  # Conv layer
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                for i in range(min(num_filters, weights.shape[0])):
                    ax = axes[i // 4, i % 4]
                    # Average across input channels for visualization
                    filter_img = weights[i].mean(dim=0).cpu().numpy()
                    ax.imshow(filter_img, cmap='viridis')
                    ax.set_title(f'Filter {i}')
                    ax.axis('off')
                
                plt.suptitle(f'Learned Filters - {layer_name}')
                plt.tight_layout()
                plt.show()
    
    def generate_feature_maps(self, input_tensor, layer_name):
        """Generate and visualize feature maps for given input"""
        import matplotlib.pyplot as plt
        
        # Forward pass to generate feature maps
        _ = self.model(input_tensor)
        
        if layer_name in self.feature_maps:
            feature_map = self.feature_maps[layer_name][0]  # First sample
            
            # Visualize first 16 channels
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            for i in range(min(16, feature_map.shape[0])):
                ax = axes[i // 4, i % 4]
                ax.imshow(feature_map[i].cpu().numpy(), cmap='viridis')
                ax.set_title(f'Channel {i}')
                ax.axis('off')
            
            plt.suptitle(f'Feature Maps - {layer_name}')
            plt.tight_layout()
            plt.show()

# Example usage
model = ExampleCNN()
grad_monitor = GradientMonitor(model, clip_value=1.0)
visualizer = FeatureVisualizer(model)
visualizer.register_feature_hooks(['conv1', 'conv2'])

# Training step
sample_input = torch.randn(4, 3, 32, 32)
sample_target = torch.randint(0, 10, (4,))

output = model(sample_input)
loss = F.cross_entropy(output, sample_target)
loss.backward()

# Check gradient summary
grad_summary = grad_monitor.get_gradient_summary()
print("Gradient Summary:", grad_summary)

# Visualize features
visualizer.visualize_filters('conv1')
visualizer.generate_feature_maps(sample_input[:1], 'conv1')
```

### Distributed Training

#### Data Parallel Training
```python
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class DistributedTrainer:
    """Distributed training setup for multi-GPU training"""
    
    def __init__(self, rank, world_size, model, train_dataset, val_dataset):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Initialize distributed training
        self.setup_distributed()
        
        # Setup model and data
        self.setup_model()
        self.setup_data()
    
    def setup_distributed(self):
        """Initialize distributed training"""
        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # Use 'gloo' for CPU training
            rank=self.rank,
            world_size=self.world_size
        )
        
        # Set device
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f'cuda:{self.rank}')
    
    def setup_model(self):
        """Setup distributed model"""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[self.rank])
    
    def setup_data(self):
        """Setup distributed data loading"""
        # Create distributed samplers
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=32,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=64,
            sampler=self.val_sampler,
            num_workers=4,
            pin_memory=True
        )
    
    def train_epoch(self, epoch, optimizer, criterion):
        """Train for one epoch with distributed setup"""
        self.model.train()
        self.train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if self.rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Gather losses from all processes
        avg_loss = total_loss / num_batches
        loss_tensor = torch.tensor(avg_loss).to(self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / self.world_size
        
        return avg_loss
    
    def validate(self, criterion):
        """Validate with distributed setup"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        # Gather metrics from all processes
        metrics = torch.tensor([total_loss, correct, total]).to(self.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        avg_loss = metrics[0].item() / self.world_size
        accuracy = metrics[1].item() / metrics[2].item()
        
        return avg_loss, accuracy
    
    def cleanup(self):
        """Clean up distributed training"""
        dist.destroy_process_group()

def run_distributed_training(rank, world_size, model_class, train_dataset, val_dataset):
    """Run distributed training on single process"""
    
    # Create model
    model = model_class()
    
    # Create trainer
    trainer = DistributedTrainer(rank, world_size, model, train_dataset, val_dataset)
    
    # Setup optimizer and criterion
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(epoch, optimizer, criterion)
        val_loss, val_acc = trainer.validate(criterion)
        
        if rank == 0:  # Only print from main process
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Cleanup
    trainer.cleanup()

# Launch distributed training
def launch_distributed_training():
    """Launch distributed training across multiple GPUs"""
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Distributed training requires at least 2 GPUs")
        return
    
    # Set environment variables
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Spawn processes
    mp.spawn(
        run_distributed_training,
        args=(world_size, ExampleCNN, train_dataset, val_dataset),
        nprocs=world_size,
        join=True
    )

# Simple DataParallel (easier but less efficient)
def simple_data_parallel_training():
    """Simple data parallel training using nn.DataParallel"""
    model = ExampleCNN()
    
    # Wrap model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.cuda()
    
    # Training as usual
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop (same as single GPU)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### Mixed Precision Training

#### Automatic Mixed Precision (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """Trainer with automatic mixed precision for faster training"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.scaler = GradScaler()
    
    def train_step(self, data, target, optimizer, criterion):
        """Single training step with mixed precision"""
        data, target = data.to(self.device), target.to(self.device)
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = self.model(data)
            loss = criterion(output, target)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Optimizer step with gradient scaling
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch with mixed precision"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = self.train_step(data, target, optimizer, criterion)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')
        
        return total_loss / num_batches
    
    def validate(self, val_loader, criterion):
        """Validation with mixed precision"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass with autocast
                with autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

# Example usage
model = ExampleCNN()
trainer = MixedPrecisionTrainer(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training with mixed precision
for epoch in range(10):
    train_loss = trainer.train_epoch(train_loader, optimizer, criterion, epoch)
    val_loss, val_acc = trainer.validate(val_loader, criterion)
    
    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
```

### Quantization

#### Post-Training Quantization
```python
import torch.quantization as quantization

class ModelQuantizer:
    """Quantize models for faster inference"""
    
    @staticmethod
    def post_training_quantization(model, calibration_loader):
        """Apply post-training quantization"""
        # Set model to evaluation mode
        model.eval()
        
        # Fuse modules for better quantization
        model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
        
        # Set quantization config
        model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model_fused)
        
        # Calibrate with representative data
        with torch.no_grad():
            for data, _ in calibration_loader:
                model_prepared(data)
        
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized
    
    @staticmethod
    def dynamic_quantization(model):
        """Apply dynamic quantization (simpler, no calibration needed)"""
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # Layers to quantize
            dtype=torch.qint8
        )
        return quantized_model
    
    @staticmethod
    def compare_model_sizes(original_model, quantized_model):
        """Compare model sizes and inference speed"""
        import time
        
        # Save models to check size
        torch.save(original_model.state_dict(), 'original_model.pth')
        torch.save(quantized_model.state_dict(), 'quantized_model.pth')
        
        import os
        original_size = os.path.getsize('original_model.pth') / 1024 / 1024  # MB
        quantized_size = os.path.getsize('quantized_model.pth') / 1024 / 1024  # MB
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        # Speed comparison
        sample_input = torch.randn(1, 3, 224, 224)
        
        # Original model speed
        original_model.eval()
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = original_model(sample_input)
        original_time = time.time() - start_time
        
        # Quantized model speed
        quantized_model.eval()
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = quantized_model(sample_input)
        quantized_time = time.time() - start_time
        
        print(f"Original model time: {original_time:.4f}s")
        print(f"Quantized model time: {quantized_time:.4f}s")
        print(f"Speed improvement: {original_time/quantized_time:.2f}x")
        
        # Clean up
        os.remove('original_model.pth')
        os.remove('quantized_model.pth')

# Example usage
model = ExampleCNN()

# Dynamic quantization (easiest)
quantized_model_dynamic = ModelQuantizer.dynamic_quantization(model)

# Post-training quantization (requires calibration data)
# quantized_model_static = ModelQuantizer.post_training_quantization(model, calibration_loader)

# Compare models
ModelQuantizer.compare_model_sizes(model, quantized_model_dynamic)
```

## Learning Objectives

By the end of this section, you should be able to:
- **Design and implement custom nn.Module classes** with proper parameter management
- **Create comprehensive training workflows** with DataLoaders, optimizers, and schedulers
- **Implement advanced training techniques** including early stopping and cross-validation
- **Utilize PyTorch ecosystem libraries** (torchvision, torchaudio, torchtext) effectively
- **Apply advanced PyTorch features** including hooks, distributed training, and quantization
- **Optimize models for production** using mixed precision and quantization techniques
- **Debug and inspect model internals** using hooks and visualization tools

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Create a custom nn.Module with proper initialization and forward pass  
□ Implement a complete training loop with validation and checkpointing  
□ Use different loss functions and optimizers appropriately  
□ Apply data augmentation and preprocessing transformations  
□ Implement early stopping and learning rate scheduling  
□ Use hooks to inspect model internals and gradients  
□ Set up distributed training across multiple GPUs  
□ Apply mixed precision training for faster computation  
□ Quantize models for efficient inference  
□ Integrate pre-trained models from torchvision  
□ Process audio data using torchaudio  
□ Handle text data for NLP tasks  

### Practical Exercises

**Exercise 1: Custom ResNet Implementation**
```python
# TODO: Implement a ResNet-18 from scratch using nn.Module
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomResNet18, self).__init__()
        # Your implementation here
        pass
    
    def forward(self, x):
        # Your implementation here
        pass

# Requirements:
# - Use proper residual connections
# - Include batch normalization
# - Support different input sizes
# - Test with CIFAR-10 dataset
```

**Exercise 2: Multi-Task Learning**
```python
# TODO: Create a model that can perform both classification and regression
class MultiTaskLearner(nn.Module):
    def __init__(self, backbone_features=512):
        super(MultiTaskLearner, self).__init__()
        # Your implementation here
        pass
    
    def forward(self, x, task_type='both'):
        # Your implementation here
        pass

# Requirements:
# - Shared backbone for feature extraction
# - Separate heads for classification and regression
# - Custom loss function combining both tasks
# - Proper training loop handling both outputs
```

**Exercise 3: Audio Classification Pipeline**
```python
# TODO: Build an audio classification system using torchaudio
class AudioClassifier:
    def __init__(self, model_path=None):
        # Your implementation here
        pass
    
    def preprocess_audio(self, file_path):
        # Load, resample, and extract features
        pass
    
    def train(self, train_loader, val_loader, num_epochs=50):
        # Complete training pipeline
        pass
    
    def predict(self, audio_file):
        # Single file prediction
        pass

# Requirements:
# - Use mel-spectrograms as input features
# - Implement data augmentation for audio
# - Support multiple audio formats
# - Include evaluation metrics specific to audio
```

**Exercise 4: Transfer Learning with Fine-tuning**
```python
# TODO: Implement a flexible transfer learning framework
class TransferLearningFramework:
    def __init__(self, base_model_name, num_classes):
        # Your implementation here
        pass
    
    def freeze_backbone(self, freeze=True):
        # Freeze/unfreeze backbone parameters
        pass
    
    def progressive_unfreezing(self, train_loader, val_loader):
        # Implement progressive unfreezing strategy
        pass
    
    def fine_tune(self, train_loader, val_loader, strategy='full'):
        # Different fine-tuning strategies
        pass

# Requirements:
# - Support multiple pre-trained models
# - Implement different unfreezing strategies
# - Compare performance of different approaches
# - Include proper evaluation and visualization
```

**Exercise 5: Model Optimization Pipeline**
```python
# TODO: Create a comprehensive model optimization pipeline
class ModelOptimizer:
    def __init__(self, model):
        self.model = model
    
    def profile_model(self, sample_input):
        # Profile memory usage and inference time
        pass
    
    def apply_quantization(self, calibration_data):
        # Apply post-training quantization
        pass
    
    def optimize_for_mobile(self):
        # Prepare model for mobile deployment
        pass
    
    def benchmark_optimizations(self, test_data):
        # Compare different optimization techniques
        pass

# Requirements:
# - Implement multiple optimization techniques
# - Measure accuracy vs. speed trade-offs
# - Generate comprehensive comparison reports
# - Support different deployment targets
```

## Study Materials

### Recommended Reading
- **Primary:** "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann
- **Advanced:** "Programming PyTorch for Deep Learning" by Ian Pointer
- **Reference:** Official PyTorch Documentation and Tutorials
- **Research:** PyTorch Papers and Technical Reports

### Online Resources
- **Official:** [PyTorch Tutorials](https://pytorch.org/tutorials/) - Comprehensive official tutorials
- **Video:** "PyTorch for Deep Learning" - FastAI Course
- **Practice:** [PyTorch Examples](https://github.com/pytorch/examples) - Official example implementations
- **Community:** PyTorch Forums and Discord

### Hands-on Labs
- **Lab 1:** Build a complete image classification pipeline from scratch
- **Lab 2:** Implement attention mechanisms and transformer architectures
- **Lab 3:** Create a multi-modal model combining vision and text
- **Lab 4:** Optimize models for edge deployment

### Technical Questions

**Architecture Questions:**
1. How does PyTorch's dynamic computation graph differ from static graphs?
2. When should you use `nn.ModuleList` vs `nn.Sequential` vs `nn.ModuleDict`?
3. How do PyTorch hooks work and what are their use cases?
4. What are the trade-offs between different optimization techniques?

**Implementation Questions:**
5. How do you implement custom autograd functions in PyTorch?
6. What's the difference between `torch.no_grad()` and `model.eval()`?
7. How do you handle variable-length sequences in PyTorch?
8. What are the best practices for memory management in PyTorch?

**Advanced Questions:**
9. How does PyTorch's distributed training work under the hood?
10. What are the considerations for mixed precision training?
11. How do you implement custom loss functions with proper gradients?
12. What are the best practices for model deployment and serving?

### Coding Challenges

**Challenge 1: Memory-Efficient Training**
```python
# Implement gradient checkpointing for memory-efficient training
class MemoryEfficientModel(nn.Module):
    def __init__(self):
        # Your implementation
        pass
    
    def forward(self, x):
        # Use gradient checkpointing to reduce memory usage
        pass
```

**Challenge 2: Custom Data Pipeline**
```python
# Create a efficient data pipeline for large datasets
class EfficientDataPipeline:
    def __init__(self, data_path, transforms=None):
        # Your implementation
        pass
    
    def create_loader(self, batch_size, num_workers=4):
        # Optimize for speed and memory
        pass
```

**Challenge 3: Advanced Regularization**
```python
# Implement multiple regularization techniques
class RegularizedModel(nn.Module):
    def __init__(self):
        # Your implementation with:
        # - Dropout variants
        # - Batch normalization
        # - Layer normalization
        # - Spectral normalization
        pass
```

### Development Environment Setup

**Required Dependencies:**
```bash
# Core PyTorch installation
pip install torch torchvision torchaudio

# Ecosystem libraries
pip install transformers
pip install pytorch-lightning
pip install torchmetrics
pip install tensorboard

# Optimization tools
pip install torchstat
pip install thop  # For FLOPs calculation

# Audio/Vision specific
pip install librosa  # Audio processing
pip install opencv-python  # Computer vision
pip install albumentations  # Advanced augmentations

# Development tools
pip install wandb  # Experiment tracking
pip install optuna  # Hyperparameter optimization
```

**GPU Setup:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

**Profiling Tools:**
```python
# PyTorch profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Your training code here
    pass

# Export results
prof.export_chrome_trace("trace.json")
```

### Performance Optimization Guidelines

**Memory Optimization:**
- Use `torch.cuda.empty_cache()` to free unused memory
- Implement gradient accumulation for large batch sizes
- Use `pin_memory=True` in DataLoader for faster GPU transfer
- Apply gradient checkpointing for memory-intensive models

**Speed Optimization:**
- Enable `torch.backends.cudnn.benchmark = True` for consistent input sizes
- Use mixed precision training with autocast
- Optimize data loading with multiple workers
- Profile your code to identify bottlenecks

**Model Optimization:**
- Apply quantization for inference speed
- Use TorchScript for production deployment
- Implement model pruning for smaller models
- Consider knowledge distillation for model compression

### Next Steps

After mastering PyTorch model development, consider exploring:
- Advanced architectures (Transformers, Graph Neural Networks)
- MLOps and model deployment strategies
- Research-oriented topics in your domain of interest
- Contributing to open-source PyTorch projects

### Example: Custom nn.Module (Enhanced)
```python
import torch.nn as nn
import torch.nn.functional as F

class AdvancedModel(nn.Module):
    """
    Enhanced example demonstrating best practices in PyTorch model development
    """
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10, 
                 dropout=0.3, use_batch_norm=True):
        super(AdvancedModel, self).__init__()
        
        # Store configuration
        self.config = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'num_classes': num_classes,
            'dropout': dropout,
            'use_batch_norm': use_batch_norm
        }
        
        # Build layers dynamically
        layer_sizes = [input_size] + hidden_sizes
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using best practices"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass with proper shape handling"""
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropouts[i](x)
        
        # Output layer (no activation)
        x = self.output_layer(x)
        
        return x
    
    def get_num_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def summary(self):
        """Print model summary"""
        total, trainable = self.get_num_parameters()
        print(f"Model Configuration: {self.config}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        return total, trainable

# Usage example
model = AdvancedModel(
    input_size=784,
    hidden_sizes=[512, 256, 128],
    num_classes=10,
    dropout=0.3,
    use_batch_norm=True
)

# Print model summary
model.summary()

# Test forward pass
sample_input = torch.randn(32, 784)
output = model(sample_input)
print(f"Output shape: {output.shape}")
```
