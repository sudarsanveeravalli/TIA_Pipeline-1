import argparse
import os
import joblib
import matplotlib.pyplot as plt
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours
from tiatoolbox.wsicore.wsireader import WSIReader
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
logger.info(f"Loading metadata from {args.metadata}")
metadata = joblib.load(args.metadata)
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

# Get the output path directly without looking for nested folders
output_dir_for_image = args.output_dir
logger.info(f"Segmentation results saved in: {output_dir_for_image}")

# Define the path to the instance map file
inst_map_path = os.path.join(output_dir_for_image, '0.dat')

# Check if the result file exists
if not os.path.exists(inst_map_path):
    logger.error(f"Result file not found: {inst_map_path}")
    exit(1)

# Load the segmentation results
logger.info(f"Loading segmentation results from {inst_map_path}")
nuclei_predictions = joblib.load(inst_map_path)
print (nuclei_predictions)
# Visualization (for tiles)
if args.mode == "tile":
    tile_img = plt.imread(args.input)
    overlaid_predictions = overlay_prediction_contours(
        canvas=tile_img,
        inst_dict={'inst_map': nuclei_predictions},
        draw_dot=False,
        line_thickness=2
    )
    plt.imshow(overlaid_predictions)
    plt.axis('off')
    overlay_path = os.path.join(args.output_dir, "nuclei_overlay.png")
    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
    logger.info(f"Nuclei overlay image saved at {overlay_path}")
    plt.show()

logger.info(f"Number of detected nuclei: {len(nuclei_predictions)}")

# Extracting the nucleus IDs and selecting the first one
nuc_id_list = list(nuclei_predictions.keys())
selected_nuc_id = nuc_id_list[0]
logger.info(f"Nucleus prediction structure for nucleus ID: {selected_nuc_id}")

sample_nuc = nuclei_predictions[selected_nuc_id]
sample_nuc_keys = list(sample_nuc)
logger.info(f"Keys in the output dictionary: {sample_nuc_keys}")

logger.info(
    f"Bounding box: ({sample_nuc['box'][0]}, {sample_nuc['box'][1]}, {sample_nuc['box'][2]}, {sample_nuc['box'][3]})"
)
logger.info(f"Centroid: ({sample_nuc['centroid'][0]}, {sample_nuc['centroid'][1]})")
