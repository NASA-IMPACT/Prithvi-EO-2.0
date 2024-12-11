from huggingface_hub import hf_hub_download
import huggingface_hub
import datasets
import glob 
import os 
import pandas as pd 
import shutil 
from argparse import ArgumentParser

token=os.environ["APITOKEN"]
huggingface_hub.login(token=token)

import datasets

parser = ArgumentParser()
parser.add_argument("--save_dir")
args = parser.parse_args()

save_dir = args.save_dir

dataset_name = "ibm-nasa-geospatial/hls_merra2_gppFlux"
# Load the dataset
dataset = datasets.load_dataset(dataset_name)

for case in ["train", "test"]:
# Get the list of files to download
    files_to_download = dataset[case].to_list()

    # Download the files
    for file in files_to_download:
        print(f"Downloading {file}")
        shutil.copy(file["image"]["path"], f"{save_dir}/{case}/images")
        try:
            shutil.copy(file["label"]["path"], f"{save_dir}/{case}/labels")
        except Exception:
            print("Label not found.")

