import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights


class FeatureExtractor(nn.Module):
    def __init__(self, last_layer_to_train=0):
        super(FeatureExtractor, self).__init__()
        
         # Load the ConvNeXt model backbone with pretrained weights
        self.backbone = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        self.out_features = 1024 # Number of output features from the backbone
        self.params = list()  # List to store parameters of trainable layers
        
        # Calculate the index from which layers should remain trainable
        index = len(self.backbone) - last_layer_to_train

        # Iterate over the layers in the backbone to set trainability
        for i, param in enumerate(self.backbone):
            if i >= index: # Train layers from the specified index onwards
                param.requires_grad_(True)
                self.params.append(param) # Add the trainable layer to the params list
            else: # Freeze layers before the specified index
                param.requires_grad_(False)

    def forward(self, x):
        # Forward pass through the backbone network
        return self.backbone(x)
