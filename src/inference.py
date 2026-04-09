import cv2
import os
import torch
from PIL import Image
from torchvision import transforms

from multitask_nn import MultitaskNN

# Define label names corresponding to the tasks: gender, bag, and hat classification.
label_names = ['gender', 'bag', 'hat']

# Function to preprocess an input image before passing it through the model.
def img_transform(img):
    # Define the preprocessing steps: resizing, normalization, and conversion to tensor.
    transforms_inference = transforms.Compose([transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])
                                               ])
    # Apply transformations to the input image.
    image_transformed = transforms_inference(img)
    return image_transformed

# Function to group possible label values for each task.
def group_label():
    return (['male', 'female'], # Gender labels
            ['no', 'yes'], # Bag labels
            ['no', 'yes'] # Hat labels
            )

# Function to preprocess an image loaded via OpenCV and prepare it for the model.
def load_from_cv(img, device):
    # Convert the color format from BGR (used by OpenCV) to RGB.
    color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the image to a PIL Image.
    pil_image = Image.fromarray(color_converted)
    # Apply preprocessing transformations.
    src = pil_image
    src = img_transform(src)
    # Add a batch dimension and move the tensor to the specified device.
    src = src.unsqueeze(dim=0)
    src = src.to(device)

    return src

# Function to load a pre-trained model from a saved file.
def load_model(network):
    load_path = os.path.join("./models", "model.pth")
     # Load the model state dictionary.
    network.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")),strict=False)
    print(f"Loaded model from {load_path}")

    return network

# Function to decode predictions and map them to human-readable labels.
def show_predictions(names, preds):
    # Unpack label group names.
    (genders, bag, hat) = names

    preds_correct = []

    # Apply the sigmoid activation function to the predictions.
    for pred in preds:
        preds_correct.append(torch.sigmoid(pred))

    # Map predictions to corresponding labels based on thresholds.
    gdr = 'male' if torch.max(preds_correct[0]).item() < 0.5 else 'female'
    bag = 'yes' if torch.max(preds_correct[1]).item() > 0.6 else 'no'
    hat = 'yes' if torch.max(preds_correct[2]).item() > 0.6 else 'no'

    # Organize predictions into a dictionary.
    predictions = {
        'gender': gdr,
        'bag': bag,
        'hat': hat
    }

    return predictions

# Main inference function to process an input image and generate predictions.
def inference(img):
    src = load_from_cv(img, device)
    preds = model(src)
    predictions = show_predictions(group_label(), preds)

    return predictions

# Set the device for computation (CUDA, MPS, or CPU).
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# Initialize the multitask neural network model.
model = MultitaskNN()
# Load the pre-trained model weights.
model = load_model(model)
# Set the model to evaluation mode and move it to the appropriate device.
model.eval()
model.to(device)
