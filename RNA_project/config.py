import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDB_TRAINING_DIR = os.path.join(DATA_DIR, "pdb_training")
POTENTIALS_DIR = os.path.join(DATA_DIR, "potentials")
PLOT_COMBINED = os.path.join(BASE_DIR, "potentials_combined.png")
PLOT_GRID = os.path.join(BASE_DIR, "potentials_grid.png")
