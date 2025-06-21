# Machine Learning Fundamentals

## Basic Concepts
- Supervised vs. unsupervised learning
- Classification, regression, clustering
- Training, validation, and testing
- Overfitting and regularization
- Feature engineering

## Neural Network Basics
- Neurons and activation functions
- Feedforward networks
- Loss functions
- Backpropagation
- Gradient descent

## Deep Learning Architectures
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers and attention mechanisms
- Generative models
- Transfer learning

### Example: Simple Neural Network (PyTorch)
```python
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)
```
