import os

# Key project directories.
ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
CONFIG_DIR = os.path.join(ROOT, "config")
DATA_DIR = os.path.join(ROOT, "fpl_optimiser", "data")