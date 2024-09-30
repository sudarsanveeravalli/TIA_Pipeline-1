from tiatoolbox.wsicore.wsireader import WSIReader
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Tissue Masking")
parser.add_argument('--input', type=str, help='Path to WSI file')
parser.add_argument('--output', type=str, help='Path to save tissue mask')

args = parser.parse_args()

# Load the WSI
wsi = WSIReader.open(args.input)

# Generate a tissue mask
tissue_mask = wsi.tissue_mask()

# Save the tissue mask
plt.imsave(args.output, tissue_mask)
