from feature_extractor import FeatureExtractor
from classification_head import Head
import torch.nn as nn
import torch



# This class represents the entire neural network, which includes a backbone for feature extraction
# and three separate branches (heads), one for each task: gender, bag, and hat classification.
class MultitaskNN(nn.Module):
    def __init__(self, last_layer_to_train=0):
        super(MultitaskNN, self).__init__()
         # Initialize the backbone responsible for extracting features from the input data.
        self.backbone = FeatureExtractor(last_layer_to_train=last_layer_to_train)

        # Initialize separate classification heads for each task.
        # Each head predicts the probability of the respective binary task.
        self.gender_head = Head(1, self.backbone.out_features)
        self.bag_head = Head(1, self.backbone.out_features)
        self.hat_head = Head(1, self.backbone.out_features)

        # Define the loss function for each task
        self.gender_loss = nn.BCELoss()
        self.bag_loss = nn.BCELoss()
        self.hat_loss = nn.BCELoss()

    def forward(self, x):
        # Pass the input data through the backbone to extract features.
        features = self.backbone(x)

        # Pass the extracted features to each classification head to get predictions.
        gender_pred = self.gender_head(features)
        bag_pred = self.bag_head(features)
        hat_pred = self.hat_head(features)

        return gender_pred, bag_pred, hat_pred



    def compute_loss(self, y_pred, y_true):
        # Extract true labels for each task (gender, hat, bag) from the ground truth.
        gender_label, hat_label, bag_label = y_true
        gender_pred, hat_pred, bag_pred = y_pred

        # Gender loss computation: 
        # # Create a mask to filter out invalid labels (negative values are ignored).
        mask_gender = (gender_label >= 0)

        gender_label_filtered = gender_label[mask_gender].unsqueeze(1) # Filter valid labels.
        gender_pred_filtered = gender_pred[mask_gender] # Filter corresponding predictions.

        # Compute loss only if there are valid labels.
        if mask_gender.sum() > 0:
            loss_gender = self.gender_loss(gender_pred_filtered, gender_label_filtered.float())
        else:
            loss_gender = torch.tensor(0.0, device=gender_pred.device)

        # Hat loss computation:
        # Create a mask for valid hat labels   
        mask_hat = (hat_label >= 0)

        hat_label_filtered = hat_label[mask_hat].unsqueeze(1) # Filter valid labels.
        hat_pred_filtered = hat_pred[mask_hat] # Filter corresponding predictions.

        # Compute loss only if there are valid labels.
        if mask_hat.sum() > 0:
            loss_hat = self.hat_loss(hat_pred_filtered, hat_label_filtered.float())
        else:
            loss_hat = torch.tensor(0.0, device=hat_pred.device)

        # Bag loss computation: 
        # Create a mask for valid bag labels
        mask_bag = (bag_label >= 0)

        bag_label_filtered = bag_label[mask_bag].unsqueeze(1) # Filter valid labels
        bag_pred_filtered = bag_pred[mask_bag] # Filter corresponding predictions.

        # Compute loss only if there are valid labels.
        if mask_bag.sum() > 0:
            loss_bag = self.bag_loss(bag_pred_filtered, bag_label_filtered.float())
        else:
            loss_bag = torch.tensor(0.0, device=bag_pred.device)

         # Return a list of individual losses for each task.
        return [loss_gender, loss_hat, loss_bag]
