import os
import argparse
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools import stainnorm
from tiatoolbox.data import stain_norm_target
import matplotlib.pyplot as plt
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description="Stain Normalization for a batch of WSIs")
parser.add_argument('--input', type=str, help='Path to WSI file')
parser.add_argument('--output', type=str, help='Directory to save normalized WSI')
parser.add_argument('--method', type=str, default='Vahadane', help='Stain normalization method (e.g., Vahadane, Macenko, Reinhard, Ruifrok)')
parser.add_argument('--extract_patch', type=bool, default=True, help='Whether to extract a patch or normalize the whole slide')

args = parser.parse_args()

# Create output directory based on WSI filename
wsi_filename = os.path.basename(args.input).split('.')[0]
output_dir = os.path.join(args.output, wsi_filename)
os.makedirs(output_dir, exist_ok=True)

# Load the WSI
wsi = WSIReader.open(args.input)

# Extract a tissue patch to normalize or process the whole WSI
if args.extract_patch:
    # Extract a region of interest (ROI) from the WSI
    sample_patch = wsi.read_region(location=[800, 1600], level=0, size=[800, 800])
else:
    # Use the entire WSI thumbnail if no patch extraction is required
    sample_patch = wsi.slide_thumbnail(1.0)

# Ensure the patch is writable
if not sample_patch.flags.writeable:
    sample_patch = np.copy(sample_patch)
    sample_patch.flags.writeable = True

# Load the target image for stain normalization (from TIAToolbox dataset)
target_image = stain_norm_target()

# Get the stain normalizer method
stain_normalizer = stainnorm.get_normalizer(args.method)
stain_normalizer.fit(target_image)

# Apply normalization to the WSI patch or WSI
normalized_image = stain_normalizer.transform(sample_patch)

# Save the normalized WSI patch or WSI
normalized_path = os.path.join(output_dir, f"normalized_wsi_{args.method}.png")
plt.imsave(normalized_path, normalized_image)

# Optionally, visualize the results (not necessary for batch processing but useful for debugging)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(target_image)
plt.title("Target Image")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(sample_patch)
plt.title("Source Image")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(normalized_image)
plt.title(f"{args.method} Stain Normalized")
plt.axis("off")
plt.show()
