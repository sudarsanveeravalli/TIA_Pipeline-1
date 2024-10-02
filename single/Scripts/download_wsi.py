from tiatoolbox import data
import os

# Directory to save the downloaded WSI
save_dir = '/home/ubuntu/bala/bala/ImpartLabs/tmp/'
os.makedirs(save_dir, exist_ok=True)

# URL to download the sample WSI
url = "https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/CMU-1-Small-Region.svs"

# Download and save the WSI
wsi_path = os.path.join(save_dir, 'CMU-1-Small-Region.svs')
data.download_data(url, wsi_path)

print(f"WSI saved at: {wsi_path}")
