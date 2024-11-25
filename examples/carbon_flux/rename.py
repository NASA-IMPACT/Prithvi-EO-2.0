import os
import glob

path = "/home/jalmeida/Datasets/carbon_flux/train/images"

files = glob.glob(os.path.join(path, "*.tiff"))

for ff in files:
    os.rename(ff, ff[:-1])
