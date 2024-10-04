import argparse
import os
import joblib
import matplotlib.pyplot as plt
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours
from image_conversion import convert_png_to_tiff, load_metadata

# Parsing input arguments
parser = argparse.ArgumentParser(description="Nuclei Segmentation using HoVerNet")
parser.add_argument('--input', type=str, help='Path to normalized WSI or tile image', required=True)
parser.add_argument('--output_dir', type=str, help='Directory to save output results', required=True)
parser.add_argument('--metadata', type=str, help='Path to metadata.pkl file', required=True)
parser.add_argument('--mode', type=str, default="wsi", choices=["wsi", "tile"], help='Processing mode: "wsi" or "tile"')
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--default_mpp', type=float, help="Default MPP to use if not found in metadata", default=0.5)
args = parser.parse_args()

# Ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Load metadata
metadata = load_metadata(args.metadata)

# Get MPP (Microns Per Pixel) from metadata or use default
mpp = metadata.get('mpp', (args.default_mpp, args.default_mpp))

# Ensure MPP is valid
if not isinstance(mpp, tuple) or len(mpp) != 2 or not all(isinstance(x, (int, float)) for x in mpp):
    mpp = (args.default_mpp, args.default_mpp)
    print(f"Invalid MPP in metadata, using default MPP: {mpp}")
else:
    print(f"Microns per pixel (MPP) from metadata: {mpp}")

# Convert PNG to TIFF before segmentation
tiff_input_path = args.input.replace('.png', '.tiff')
tiff_input = convert_png_to_tiff(args.input, tiff_input_path, metadata=metadata)

# Initialize NucleusInstanceSegmentor
segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",  # Use "hovernet_original-kumar" if needed
    num_loader_workers=2,
    num_postproc_workers=2,
    batch_size=4,
    auto_generate_mask=False
)

# Run the segmentation
print(f"Running segmentation on {tiff_input}")
output = segmentor.predict(
    imgs=[tiff_input],
    save_dir=args.output_dir,
    mode=args.mode,
    on_gpu=args.gpu,
    crash_on_exception=True
)

# Load the segmentation results
result_file = output[0][1]
nuclei_predictions = joblib.load(result_file)

# Visualization for tile mode
if args.mode == "tile":
    # Load the input TIFF image
    tile_img = plt.imread(tiff_input)

    # Define color mapping for visualization
    color_dict = {
        0: ("background", (255, 165, 0)),
        1: ("neoplastic epithelial", (255, 0, 0)),
        2: ("Inflammatory", (255, 255, 0)),
        3: ("Connective", (0, 255, 0)),
        4: ("Dead", (0, 0, 0)),
        5: ("non-neoplastic epithelial", (0, 0, 255)),
    }

    # Overlay the predictions on the original image
    overlaid_predictions = overlay_prediction_contours(
        canvas=tile_img,
        inst_dict=nuclei_predictions,
        draw_dot=False,
        type_colours=color_dict,
        line_thickness=2
    )

    # Save and show the overlaid image
    plt.imshow(overlaid_predictions)
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, "nuclei_overlay.png"), bbox_inches='tight', pad_inches=0)
    plt.show()

print(f"Segmentation completed. Results saved in {args.output_dir}")
