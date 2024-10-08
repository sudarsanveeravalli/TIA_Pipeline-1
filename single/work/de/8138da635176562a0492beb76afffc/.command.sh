#!/bin/bash -ue
python /home/ubuntu/bala/bala/ImpartLabs/TIA_Pipeline/single/Scripts/hovernet.py --input normalized_wsi.png --mask tissue_mask.png --output nuclei_result.pkl
