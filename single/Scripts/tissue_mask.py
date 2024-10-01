import os
import argparse
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.tissuemask import MorphologicalMasker
import matplotlib.pyplot as plt
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description="Tissue Masking for WSIs")
parser.add_argument('--input', type=str, help='Path to WSI file')
parser.add_argument('--output', type=str, help='Directory to save tissue mask')
parser.add_argument('--resolution', type=float, default=1.25, help='Resolution for tissue mask generation (in objective power)')

args = parser.parse_args()

# Create output directory based on WSI filename
wsi_filename = os.path.basename(args.input).split('.')[0]
output_dir = os.path.join(args.output, wsi_filename)
os.makedirs(output_dir, exist_ok=True)

# Load the WSI
wsi = WSIReader.open(args.input)

# Generate the tissue mask
mask = wsi.tissue_mask(resolution=args.resolution, units="power")

# Generate thumbnail for visualization purposes
wsi_thumb = wsi.slide_thumbnail(resolution=args.resolution, units="power")
mask_thumb = mask.slide_thumbnail(resolution=args.resolution, units="power")

# Ensure the mask is writable if needed (in case it's read-only)
if not mask_thumb.flags.writeable:
    mask_thumb = np.copy(mask_thumb)
    mask_thumb.flags.writeable = True

# Save the tissue mask thumbnail
mask_path = os.path.join(output_dir, "tissue_mask.png")
plt.imsave(mask_path, mask_thumb)

# Optionally, visualize the results (useful for debugging, not necessary in batch mode)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(wsi_thumb)
plt.title("WSI Thumbnail")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask_thumb)
plt.title("Tissue Mask")
plt.axis("off")

plt.savefig(os.path.join(output_dir, "mask_visualization.png"))
plt.show()
