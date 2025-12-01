#!/usr/bin/env python3
"""
run_pipeline.py
"""

import subprocess
import sys
import os
import config

# Executables
SCRIPT_TRAIN = "train_potential.py"
SCRIPT_PLOT = "plot_potentials.py"
SCRIPT_SCORE = "score_structure.py"


def run_command(command):
    """Helper to run a shell command and stop if it fails."""
    print(f"\n[Pipeline] Running: {' '.join(command)}")
    try:
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
        config.PDB_TRAINING_DIR,
        "--out_dir",
        config.POTENTIALS_DIR,
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
        config.POTENTIALS_DIR,
        "--out_combined",
        config.PLOT_COMBINED,
        "--out_grid",
        config.PLOT_GRID,
    ]
    run_command(cmd_plot)

    # 3. SCORING
    # ---------------------------------------------------------
    print("\n--- Step 3: Scoring ---")
    cmd_score = [
        sys.executable,
        SCRIPT_SCORE,
        "--pdb_dir",
        config.PDB_TRAINING_DIR,
        "--pot_dir",
        config.POTENTIALS_DIR,
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
