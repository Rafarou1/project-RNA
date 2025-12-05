#!/usr/bin/env python3
"""
run_pipeline.py

Run the full RNA statistical potential training, plotting, and scoring pipeline.
"""

import subprocess
import sys
import os
import argparse
import config

# Executables
SCRIPT_TRAIN = "train_potential.py"
SCRIPT_PLOT = "plot_potentials.py"
SCRIPT_SCORE = "score_structure.py"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the full RNA potential pipeline.")

    # Optional arguments to override defaults
    parser.add_argument(
        "--out_dir",
        default=config.POTENTIALS_DIR,
        help=f"Directory to store results (default: {config.POTENTIALS_DIR})",
    )
    parser.add_argument("--atom", default="C3'", help="Atom type to use (default: C3')")
    parser.add_argument(
        "--max_dist", type=float, default=20.0, help="Max distance in Å (default: 20.0)"
    )
    parser.add_argument(
        "--bin_width", type=float, default=1.0, help="Bin width in Å (default: 1.0)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress."
    )

    return parser.parse_args()


def run_command(command):
    """Helper to run a shell command and stop if it fails."""
    print(f"\n[Pipeline] Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Step failed with exit code {e.returncode}")
        sys.exit(1)


def run_full_pipeline(args):
    print("=== Starting RNA Potential Pipeline ===")
    print(f"Output Directory: {args.out_dir}")
    print(f"Settings: Atom={args.atom}, Max={args.max_dist}A, Width={args.bin_width}A")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created directory: {args.out_dir}")

    # 1. TRAINING
    print("\n--- Step 1: Training ---")
    cmd_train = [
        sys.executable,
        SCRIPT_TRAIN,
        "--pdb_dir",
        config.PDB_TRAINING_DIR,
        "--out_dir",
        args.out_dir,
        "--atom",
        args.atom,
        "--max_dist",
        str(args.max_dist),
        "--bin_width",
        str(args.bin_width),
    ]
    if args.verbose:
        cmd_train.append("--verbose")

    run_command(cmd_train)

    # 2. PLOTTING
    print("\n--- Step 2: Plotting ---")

    dir_name = os.path.basename(os.path.normpath(args.out_dir))
    plot_combined = os.path.join(args.out_dir, f"combined_{dir_name}.png")
    plot_grid = os.path.join(args.out_dir, f"grid_{dir_name}.png")
    plot_hist = os.path.join(args.out_dir, f"histograms_{dir_name}.png")

    cmd_plot = [
        sys.executable,
        SCRIPT_PLOT,
        "--in_dir",
        args.out_dir,
        "--out_combined",
        plot_combined,
        "--out_grid",
        plot_grid,
        "--out_hist",
        plot_hist,
    ]
    run_command(cmd_plot)

    # 3. SCORING
    print("\n--- Step 3: Scoring ---")
    cmd_score = [
        sys.executable,
        SCRIPT_SCORE,
        "--pdb_dir",
        config.PDB_TRAINING_DIR,
        "--pot_dir",
        args.out_dir,
    ]
    run_command(cmd_score)

    print("\n=== Pipeline Completed Successfully ===")
    print(f"Results stored in: {args.out_dir}")


if __name__ == "__main__":
    # Simple check to ensure scripts exist
    for s in [SCRIPT_TRAIN, SCRIPT_PLOT, SCRIPT_SCORE]:
        if not os.path.exists(s):
            print(f"Error: Script {s} not found in current directory.")
            sys.exit(1)

    args = parse_arguments()
    run_full_pipeline(args)
