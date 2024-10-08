import argparse
import os
import joblib
import matplotlib.pyplot as plt
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours
from tiatoolbox.wsicore.wsireader import WSIReader
from skimage import measure  # Use skimage to compute region properties
import logging
import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Parsing input arguments
parser = argparse.ArgumentParser(description="Nuclei Segmentation using HoVerNet")
parser.add_argument('--input', type=str, help='Path to normalized image or WSI', required=True)
parser.add_argument('--output_dir', type=str, help='Directory to save output results', required=True)
parser.add_argument('--metadata', type=str, help='Path to metadata.pkl file', required=False)
parser.add_argument('--mode', type=str, default="tile", choices=["wsi", "tile"], help='Processing mode: "wsi" or "tile"')
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--default_mpp', type=float, help="Default MPP if not found in metadata", default=0.5)
args = parser.parse_args()

if not args.gpu:
    args.gpu = torch.cuda.is_available()
logger.info(f"Using GPU for processing: {args.gpu}")

logger.debug(f"Input arguments: {args}")

# Load metadata
if args.metadata:
    logger.info(f"Loading metadata from {args.metadata}")
    metadata = joblib.load(args.metadata)
else:
    metadata = {}
    logger.warning("No metadata provided, using default MPP.")

mpp = metadata.get('mpp', args.default_mpp)

# Ensure MPP is valid and calculate a single MPP value
if isinstance(mpp, tuple) and len(mpp) == 2:
    mpp_value = sum(mpp) / len(mpp)
elif isinstance(mpp, (int, float)):
    mpp_value = float(mpp)
else:
    mpp_value = args.default_mpp
    logger.warning(f"Invalid MPP in metadata, using default MPP: {mpp_value}")

logger.info(f"Microns per pixel (MPP) used: {mpp_value}")

# Initialize NucleusInstanceSegmentor
logger.info("Initializing NucleusInstanceSegmentor")
segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",
    num_loader_workers=2,
    num_postproc_workers=2,
    batch_size=4,
    auto_generate_mask=False
)

# Process depending on the mode (wsi or tile)
if args.mode == "wsi":
    logger.info(f"Running segmentation on WSI: {args.input}")
    try:
        wsi = WSIReader.open(args.input)
        output = segmentor.predict(
            imgs=[wsi],
            save_dir=args.output_dir,
            mode='wsi',
            on_gpu=args.gpu,
            crash_on_exception=False
        )
    except Exception as e:
        logger.error(f"Segmentation failed for WSI: {e}")
        exit(1)
else:
    logger.info(f"Running segmentation on Tile: {args.input}")
    try:
        output = segmentor.predict(
            imgs=[args.input],
            save_dir=args.output_dir,
            mode='tile',
            on_gpu=args.gpu,
            crash_on_exception=False
        )
    except Exception as e:
        logger.error(f"Segmentation failed for Tile: {e}")
        exit(1)

logger.debug(f"Segmentation output: {output}")

# Get the correct path to the output file
output_dir_for_image = args.output_dir
logger.info(f"Segmentation results saved in: {output_dir_for_image}")

# Define the path to the instance map file
inst_map_path = os.path.join(output_dir_for_image, '0.dat')



# Load the segmentation results
logger.info(f"Loading segmentation results from {inst_map_path}")
nuclei_predictions = joblib.load(inst_map_path)


# Debug: Inspect the nuclei_predictions
logger.debug(f"nuclei_predictions type: {type(nuclei_predictions)}")
logger.debug(f"nuclei_predictions content: {nuclei_predictions}")

# Visualization (for tiles)
if args.mode == "tile":
    tile_img = plt.imread(args.input)

    # Load the instance map
    inst_map = nuclei_predictions.get('inst_map', None)
    if inst_map is None:
        logger.error("Instance map not found in nuclei predictions.")
        exit(1)

    # Generate instance data including contours using skimage
    regions = measure.regionprops(inst_map)
    inst_data = []

    for region in regions:
        # Get the contour coordinates
        contours = measure.find_contours(inst_map == region.label, 0.5)
        contour = contours[0] if contours else None
        inst_data.append({
            'label': region.label,
            'bbox': region.bbox,
            'centroid': region.centroid,
            'contour': contour,
        })

    inst_dict = {'inst_map': inst_map, 'instances': inst_data}

    overlaid_predictions = overlay_prediction_contours(
        canvas=tile_img,
        inst_dict=inst_dict,
        draw_dot=False,
        line_thickness=2
    )

    plt.imshow(overlaid_predictions)
    plt.axis('off')
    overlay_path = os.path.join(args.output_dir, "nuclei_overlay.png")
    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
    logger.info(f"Nuclei overlay image saved at {overlay_path}")
    plt.show()

    logger.info(f"Number of detected nuclei with contours: {len(inst_data)}")

    # Extracting the nucleus IDs and selecting the first one
    if inst_data:
        selected_nuc = inst_data[0]  # inst_data is a list of dicts
        selected_nuc_id = selected_nuc['label']
        logger.info(f"Nucleus prediction structure for nucleus ID: {selected_nuc_id}")

        sample_nuc_keys = list(selected_nuc.keys())
        logger.info(f"Keys in the output dictionary: {sample_nuc_keys}")

        bbox = selected_nuc['bbox']  # (min_row, min_col, max_row, max_col)
        logger.info(
            f"Bounding box: ({bbox[1]}, {bbox[0]}, {bbox[3]}, {bbox[2]})"
        )
        centroid = selected_nuc['centroid']  # (row, col)
        logger.info(f"Centroid: ({centroid[1]}, {centroid[0]})")
    else:
        logger.warning("No valid nuclei found with contours.")

logger.info("Processing complete.")
