from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.data import stain_norm_target
import argparse
import matplotlib.pyplot as plt

# Argument parser
parser = argparse.ArgumentParser(description="Stain Normalization")
parser.add_argument('--input', type=str, help='Path to WSI file')
parser.add_argument('--output', type=str, help='Path to save normalized WSI')

args = parser.parse_args()

# Load the WSI
wsi = WSIReader.open(args.input)

# Extract a tissue patch to normalize (resize for faster computation)
tissue_patch = wsi.slide_thumbnail(1.0)

# Load the target image for stain normalization (predefined target from TIAToolbox)
target_image = stain_norm_target()

# Get the stain normalizer (Vahadane)
normalizer = get_normalizer(method="vahadane")

# Fit the normalizer to the target image
normalizer.fit(target_image)

# Apply normalization to the WSI patch
normalized_image = normalizer.transform(tissue_patch)

# Save the normalized WSI patch
plt.imsave(args.output, normalized_image)
