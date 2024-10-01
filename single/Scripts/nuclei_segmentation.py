import argparse
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.visualization import overlay_prediction_contours
import joblib
import matplotlib.pyplot as plt

# Parsing input arguments
parser = argparse.ArgumentParser(description="Nuclei Segmentation using HoVerNet")
parser.add_argument('--input', type=str, help='Path to normalized WSI or tile image')
parser.add_argument('--output_dir', type=str, help='Directory to save output results')
parser.add_argument('--mode', type=str, default="wsi", choices=["wsi", "tile"], help='Processing mode: "wsi" or "tile"')
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
args = parser.parse_args()

# Initialize NucleusInstanceSegmentor
segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",  # Use "hovernet_original-kumar" if needed
    num_loader_workers=2,
    num_postproc_workers=2,
    batch_size=4,
    auto_generate_mask=False
)

# Run the segmentation
print(f"Running segmentation on {args.input}")
output = segmentor.predict(
    imgs=[args.input],
    save_dir=args.output_dir,
    mode=args.mode,
    on_gpu=args.gpu,
    crash_on_exception=True
)

# Load the segmentation results (dictionary with instance segmentation details)
result_file = f"{output[0][1]}.dat"
nuclei_predictions = joblib.load(result_file)

# If desired, visualize one of the results (for tiles)
if args.mode == "tile":
    # Load the input image
    tile_img = plt.imread(args.input)

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

    # Show and save the overlaid image
    plt.imshow(overlaid_predictions)
    plt.axis('off')
    plt.savefig(f"{args.output_dir}/nuclei_overlay.png")
    plt.show()

print(f"Segmentation completed. Results saved in {args.output_dir}")
