import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDB_TRAINING_DIR = os.path.join(DATA_DIR, "pdb_training")
POTENTIALS_DIR = os.path.join(DATA_DIR, "potentials_multi")
PLOT_COMBINED = os.path.join(POTENTIALS_DIR, "potentials_combined_multi.png")
PLOT_GRID = os.path.join(POTENTIALS_DIR, "potentials_grid_multi.png")