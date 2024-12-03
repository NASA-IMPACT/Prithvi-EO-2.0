import os
import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()

path = args.path

files = glob.glob(os.path.join(path, "*.tiff"))

for ff in files:
    os.rename(ff, ff[:-1])
