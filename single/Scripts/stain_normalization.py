#!/usr/bin/env python 

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
from tiatoolbox import data, logger
from tiatoolbox.tools import stainnorm
from tiatoolbox.wsicore.wsireader import WSIReader

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Stain Normalization with TIFF output")
parser.add_argument('--input', type=str, required=True, help='Path to input WSI file')
parser.add_argument('--output', type=str, required=True, help='Path to save normalized WSI image as TIFF')
parser.add_argument('--reference', type=str, help='Path to reference image for stain normalization', default=None)
parser.add_argument('--method', type=str, choices=['vahadane', 'macenko', 'reinhard', 'ruifrok'], default='vahadane', help='Stain normalization method to use')

args = parser.parse_args()

# Set up logging
logger.setLevel('INFO')

# Load the WSI and extract metadata
wsi_reader = WSIReader.open(args.input)
metadata = wsi_reader.info.as_dict()  # Save metadata to reapply later

# Extract full-resolution WSI or appropriate resolution based on requirements
slide_image = wsi_reader.read_region(location=(0, 0), level=0, size=wsi_reader.slide_dimensions(resolution=0.5, units="mpp")[0])

# Create a writable copy of the image
slide_image_writable = np.array(slide_image)  # Convert to NumPy array
if not slide_image_writable.flags.writeable:
    slide_image_writable = np.copy(slide_image_writable)  # Ensure writable

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

# Perform stain normalization on the slide image
normalized_image = stain_normalizer.transform(slide_image_writable)

# Ensure the output directory exists
output_path = Path(args.output)
output_dir = output_path.parent
output_dir.mkdir(parents=True, exist_ok=True)

# Convert to RGB format if necessary (to ensure compatibility with TIFF format)
if normalized_image.shape[-1] == 4:  # RGBA to RGB if needed
    normalized_image = normalized_image[:, :, :3]

# Convert normalized_image (NumPy array) to PIL Image for saving as TIFF
normalized_image_pil = Image.fromarray((normalized_image * 255).astype(np.uint8))

# Define resolution metadata based on MPP (Microns Per Pixel)
x_resolution = 1 / metadata.get('mpp', [0.5])[0]  # Default to 0.5 MPP if missing
y_resolution = 1 / metadata.get('mpp', [0.5])[1]

# Save the normalized image as TIFF, embedding resolution metadata
normalized_image_pil.save(output_path, resolution=(x_resolution, y_resolution), dpi=(x_resolution, y_resolution))

logger.info(f"Stain normalization completed. Normalized TIFF image saved to {output_path}")
