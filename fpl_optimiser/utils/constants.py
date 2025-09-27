import os

# Key project directories.
ROOT_RAW = os.path.join(os.path.dirname(__file__), "..", "..")

ROOT = os.path.normpath(os.path.abspath(ROOT_RAW))
CONFIG_DIR = os.path.join(ROOT, "config")
DATA_DIR = os.path.join(ROOT, "fpl_optimiser", "data")