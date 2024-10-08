#!/bin/bash -ue
python /home/ubuntu/bala/bala/ImpartLabs/TIA_Pipeline/single/Scripts/nuclei_segmentation.py --input normalized_wsi.png --mask tissue_mask.png --output nuclei_result.pkl
