import argparse
import os
import joblib
import matplotlib.pyplot as plt
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Parsing input arguments
parser = argparse.ArgumentParser(description="Nuclei Segmentation using HoVerNet")
parser.add_argument('--input', type=str, help='Path to normalized image', required=True)
parser.add_argument('--output_dir', type=str, help='Directory to save output results', required=True)
parser.add_argument('--metadata', type=str, help='Path to metadata.pkl file', required=True)
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--default_mpp', type=float, help="Default MPP if not found in metadata", default=0.5)
args = parser.parse_args()

# Load metadata
metadata = joblib.load(args.metadata)
mpp = metadata.get('mpp', args.default_mpp)

# Ensure MPP is valid and calculate a single MPP value
if isinstance(mpp, tuple) and len(mpp) == 2:
    mpp_value = sum(mpp) / len(mpp)
elif isinstance(mpp, (int, float)):
    mpp_value = float(mpp)
else:
    mpp_value = args.default_mpp
    print(f"Invalid MPP in metadata, using default MPP: {mpp_value}")

print(f"Microns per pixel (MPP) used: {mpp_value}")

# Initialize NucleusInstanceSegmentor
segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-monusac",  # Try a different model
    num_loader_workers=2,
    num_postproc_workers=2,
    batch_size=1,
    auto_generate_mask=False
)

# Run segmentation on the original PNG image
try:
    output = segmentor.predict(
        imgs=[args.input],
        save_dir=args.output_dir,
        mode='tile',
        on_gpu=args.gpu,
        crash_on_exception=False,
        resolution=mpp_value,
        units="mpp",
    )
except Exception as e:
    print(f"Segmentation failed: {e}")
    exit(1)

# Print the output variable
print(f"Segmentation output: {output}")

# Get the output directory for the first image
output_dir_for_image = output[0][1]  # This should be a directory

# Define the path to the instance map file
inst_map_path = os.path.join(output_dir_for_image, 'inst_map.dat')

# Check if the result file exists
if not os.path.exists(inst_map_path):
    print(f"Result file not found: {inst_map_path}")
    exit(1)

# Load the segmentation results
nuclei_predictions = joblib.load(inst_map_path)

# Load the input PNG image
tile_img = plt.imread(args.input)

# Visualization
overlaid_predictions = overlay_prediction_contours(
    canvas=tile_img,
    inst_dict={'inst_map': nuclei_predictions},
    draw_dot=False,
    line_thickness=2
)

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Save and display the overlaid image
plt.imshow(overlaid_predictions)
plt.axis('off')
plt.savefig(os.path.join(args.output_dir, "nuclei_overlay.png"), bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Segmentation completed. Results saved in {args.output_dir}")
