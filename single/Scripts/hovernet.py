from tiatoolbox.models.architecture.hovernet import HoVerNet
from tiatoolbox.models.engine import NucleusInstanceSegmentor
import argparse
import pickle

parser = argparse.ArgumentParser(description="HoVerNet Nuclei Segmentation")
parser.add_argument('--input', type=str, help='Path to WSI file')
parser.add_argument('--mask', type=str, help='Path to tissue mask')
parser.add_argument('--output', type=str, help='Path to save nuclei result')

args = parser.parse_args()

# Initialize HoVerNet model
hovernet_model = HoVerNet(task='nuclei')
segmentor = NucleusInstanceSegmentor(hovernet_model)

# Perform segmentation
nuclei_result = segmentor.predict([args.input])

# Save the segmentation results
with open(args.output, 'wb') as f:
    pickle.dump(nuclei_result, f)
