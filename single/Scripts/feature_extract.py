import pickle
import numpy as np
import pandas as pd
import argparse
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.engine.semantic_segmentor import WholeSlideClassifier
from tiatoolbox.utils.transforms import img_resize

parser = argparse.ArgumentParser(description="Feature Extraction using TIAToolbox ResNet")
parser.add_argument('--input', type=str, help='Path to nuclei segmentation result')
parser.add_argument('--output', type=str, help='Path to save extracted features')

args = parser.parse_args()

# Load nuclei segmentation result
with open(args.input, 'rb') as f:
    nuclei_result = pickle.load(f)

# Initialize TIAToolbox's pre-trained ResNet model for feature extraction
resnet_model = get_pretrained_model("resnet50", num_classes=2)  # Use num_classes based on your problem

# List to hold extracted features
feature_list = []

# Loop through each instance in the nuclei segmentation results
for i, instance in enumerate(nuclei_result[0].inst_pred):
    # Assuming each `instance` is a segmentation mask or image
    instance_img = img_resize(instance, (224, 224))  # Resize instance to 224x224 for ResNet input

    # Run the ResNet model to extract features
    features = resnet_model.predict(instance_img)

    # Collect features for the current instance
    feature_list.append(features.flatten())  # Flatten the feature vector

# Convert features to a DataFrame
df = pd.DataFrame(feature_list)

# Save extracted features to CSV
df.to_csv(args.output, index=False)
