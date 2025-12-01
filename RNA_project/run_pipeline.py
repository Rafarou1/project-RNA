#!/usr/bin/env python3
"""
run_pipeline.py

Orchestrates the full RNA structural bioinformatics pipeline:
1. Train potentials from a PDB dataset.
2. Plot the resulting potentials.
3. Score a specific target structure.
"""

import subprocess
import sys
import os

# --- Configuration ---
# Adjust these paths to match your project structure
DATA_DIR = "data"
PDB_TRAINING_DIR = os.path.join(DATA_DIR, "pdb_training")
POTENTIALS_DIR = os.path.join(DATA_DIR, "potentials_multi")
PLOT_FILE = os.path.join(POTENTIALS_DIR, "potentials_multi.png")

# Executables
SCRIPT_TRAIN = "train_potential.py"
SCRIPT_PLOT = "plot_potentials.py"
SCRIPT_SCORE = "score_structure.py"


def run_command(command):
    """Helper to run a shell command and stop if it fails."""
    print(f"\n[Pipeline] Running: {' '.join(command)}")
    try:
        # check=True raises an error if the command fails (non-zero exit code)
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Step failed with exit code {e.returncode}")
        sys.exit(1)


def run_full_pipeline():
    print("=== Starting RNA Potential Pipeline ===")

    # 1. TRAINING
    # ---------------------------------------------------------
    print("\n--- Step 1: Training ---")
    cmd_train = [
        sys.executable,
        SCRIPT_TRAIN,
        "--pdb_dir",
        PDB_TRAINING_DIR,
        "--out_dir",
        POTENTIALS_DIR,
        "--verbose",
    ]
    run_command(cmd_train)

    # 2. PLOTTING
    # ---------------------------------------------------------
    print("\n--- Step 2: Plotting ---")
    cmd_plot = [
        sys.executable,
        SCRIPT_PLOT,
        "--in_dir",
        POTENTIALS_DIR,
        "--out_png",
        PLOT_FILE,
    ]
    run_command(cmd_plot)

    # 3. SCORING
    # ---------------------------------------------------------
    print("\n--- Step 3: Scoring ---")
    cmd_score = [
        sys.executable,
        SCRIPT_SCORE,
        "--pdb_dir",
        PDB_TRAINING_DIR,
        "--pot_dir",
        POTENTIALS_DIR,
    ]
    run_command(cmd_score)

    print("\n=== Pipeline Completed Successfully ===")


if __name__ == "__main__":
    # Simple check to ensure scripts exist
    for s in [SCRIPT_TRAIN, SCRIPT_PLOT, SCRIPT_SCORE]:
        if not os.path.exists(s):
            print(f"Error: Script {s} not found in current directory.")
            sys.exit(1)

    run_full_pipeline()
