#!/usr/bin/env python 

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tiatoolbox import data, logger
from tiatoolbox.tools import stainnorm
from tiatoolbox.wsicore.wsireader import WSIReader

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Stain Normalization")
parser.add_argument('--input', type=str, required=True, help='Path to input WSI file')
parser.add_argument('--output', type=str, required=True, help='Path to save normalized WSI image')
parser.add_argument('--reference', type=str, help='Path to reference image for stain normalization', default=None)
parser.add_argument('--method', type=str, choices=['vahadane', 'macenko', 'reinhard', 'ruifrok'], default='vahadane', help='Stain normalization method to use')

args = parser.parse_args()

# Set up logging
logger.setLevel('INFO')

# Load the WSI and extract metadata
wsi_reader = WSIReader.open(args.input)
metadata = wsi_reader.info.as_dict()  # Save metadata to reapply later

# Extract slide at low resolution (or adjust for higher resolution)
slide_image = wsi_reader.slide_thumbnail(resolution=1.25, units="power")

# Load or set the reference image
if args.reference:
    reference_image = plt.imread(args.reference)
    logger.info(f"Using provided reference image: {args.reference}")
else:
    reference_image = data.stain_norm_target()
    logger.info("Using default reference image from tiatoolbox.")

# Initialize the stain normalizer
if args.method == 'vahadane':
    stain_normalizer = stainnorm.VahadaneNormalizer()
elif args.method == 'macenko':
    stain_normalizer = stainnorm.MacenkoNormalizer()
elif args.method == 'reinhard':
    stain_normalizer = stainnorm.ReinhardNormalizer()
elif args.method == 'ruifrok':
    stain_normalizer = stainnorm.RuifrokNormalizer()
else:
    raise ValueError(f"Unsupported stain normalization method: {args.method}")

# Fit the normalizer to the reference image
stain_normalizer.fit(reference_image)

# Make the slide image writable by copying it
slide_image_copy = slide_image.copy()

# Perform stain normalization on the copied image
normalized_image = stain_normalizer.transform(slide_image_copy)

# Ensure the output directory exists
output_path = Path(args.output)
output_dir = output_path.parent
output_dir.mkdir(parents=True, exist_ok=True)

# Save the normalized image
plt.imsave(str(output_path), normalized_image)

# Save the metadata to a file for future use
metadata_path = str(output_dir / 'metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

logger.info(f"Stain normalization completed. Normalized image saved to {output_path}")
logger.info(f"Metadata saved to {metadata_path}")
