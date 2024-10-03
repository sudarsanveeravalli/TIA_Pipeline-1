import joblib
import numpy as np
import pandas as pd
import argparse
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.utils.transforms import img_resize

parser = argparse.ArgumentParser(description="Feature Extraction using TIAToolbox ResNet")
parser.add_argument('--input', type=str, help='Path to nuclei segmentation result (0.dat)', required=True)
parser.add_argument('--output', type=str, help='Path to save extracted features', required=True)

args = parser.parse_args()

# Load nuclei segmentation result from 0.dat
nuclei_result = joblib.load(args.input)  # Load segmentation result from 0.dat

# Initialize TIAToolbox's pre-trained ResNet model for feature extraction
resnet_model = get_pretrained_model("resnet50", num_classes=2)  # Adjust num_classes based on your task

# List to hold extracted features
feature_list = []

# Loop through each instance in the nuclei segmentation results
for instance_id, instance_data in nuclei_result.items():
    # Each instance contains segmentation details; get the instance image (mask)
    instance_img = instance_data['img']  # Assuming 'img' stores the instance mask/image

    # Resize instance to 224x224 for ResNet input
    instance_img_resized = img_resize(instance_img, (224, 224))

    # Run the ResNet model to extract features
    features = resnet_model.predict(instance_img_resized)

    # Collect features for the current instance
    feature_list.append(features.flatten())  # Flatten the feature vector

# Convert features to a DataFrame
df = pd.DataFrame(feature_list)

# Save extracted features to CSV
df.to_csv(args.output, index=False)

print(f"Feature extraction completed. Features saved to {args.output}")
