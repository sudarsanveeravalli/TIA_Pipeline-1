import pickle
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Feature Extraction")
parser.add_argument('--input', type=str, help='Path to nuclei segmentation result')
parser.add_argument('--output', type=str, help='Path to save extracted features')

args = parser.parse_args()

# Load nuclei segmentation result
with open(args.input, 'rb') as f:
    nuclei_result = pickle.load(f)

# Extract features like nuclei count and density
nuclei_count = np.unique(nuclei_result[0].inst_pred).size - 1

# Save features
df = pd.DataFrame({'nuclei_count': [nuclei_count]})
df.to_csv(args.output, index=False)
