import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import CBAM

# This script defines two neural network modules for classification tasks:
# 1. **ClassificationModule**: A generic classification module with four fully connected layers, ReLU activations,
#    and a dropout layer to mitigate overfitting.
# 2. **Head**: Combines an attention mechanism (CBAM), adaptive average pooling, and the ClassificationModule.
#    It is designed to refine input feature maps using attention, reduce their dimensions, and produce class predictions.


# This class implements a classification module with four fully connected (dense) layers and dropout.
class ClassificationModule(nn.Module):
    def __init__(self, num_classes, input_features=1024):
        super(ClassificationModule, self).__init__()

        # Define the number of classes and input features.
        self.num_classes = num_classes
        self.input_features = input_features

         # Fully connected layers for progressively reducing feature dimensions.
        self.fc1 = nn.Linear(self.input_features, 512)  # first dense layer
        self.fc2 = nn.Linear(512, 256)  # second dense layer
        self.fc3 = nn.Linear(256, 128)  # third dense layer
        self.fc4 = nn.Linear(128, num_classes)  # fourth dense layer

        # Dropout layer to reduce overfitting by randomly deactivating neurons.
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Apply ReLU activation to the output of each dense layer.
        x = F.relu(self.fc1(x)) # First layer activation.
        x = F.relu(self.fc2(x)) # Second layer activation.
        x = F.relu(self.fc3(x)) # Third layer activation.
        # Apply dropout before the final dense layer.
        x = self.dropout(x)
        # Pass through the final layer to get class scores.
        x = self.fc4(x)

        return x

# This class defines a "Head" that combines attention mechanisms, pooling, and classification.
class Head(nn.Module):
    def __init__(self, num_classes, input_features=1024):
        super(Head, self).__init__()

        # Define the number of classes and input features.
        self.num_classes = num_classes
        self.input_features = input_features

        # Attention module (CBAM) to focus on important regions of the input.
        self.attention = CBAM(self.input_features)

         # Adaptive average pooling to reduce the spatial dimensions to 1x1.
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classification module for final predictions.
        self.fc = ClassificationModule(num_classes=self.num_classes, input_features=self.input_features)

    def forward(self, x):
        # Apply the attention mechanism to the input.
        x = self.attention(x)

        # Reduce the spatial dimensions using adaptive average pooling.
        x = self.avgpool(x)

         # Flatten the tensor to prepare it for the fully connected layers.
        x = x.view(x.size(0), -1)

        # Pass the flattened tensor through the classification module.
        x = self.fc(x)

        return x
