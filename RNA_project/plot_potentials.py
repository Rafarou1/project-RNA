#!/usr/bin/env python3
"""
plot_potentials.py

Plot distance-dependent potentials for RNA base pairs.

Usage:
    python plot_potentials.py --in_dir data/potentials --out_png potentials.png
"""

import os
import argparse
import matplotlib.pyplot as plt
import math
import sys
from rna_utils import load_params, PAIR_TYPES


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot RNA potentials.")
    parser.add_argument(
        "--in_dir", required=True, help="Directory containing potential files."
    )
    parser.add_argument(
        "--out_combined",
        default="potentials_combined.png",
        help="Filename for the combined plot.",
    )
    parser.add_argument(
        "--out_grid", default="potentials_grid.png", help="Filename for the grid plot."
    )
    return parser.parse_args()


def load_all_scores(in_dir, x_axis_len):
    """Pre-loads all potential files into a dictionary."""
    data = {}
    for pair in PAIR_TYPES:
        filename = os.path.join(in_dir, f"potential_{pair}.txt")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                scores = [float(line.strip()) for line in f if line.strip()]

            if len(scores) == x_axis_len:
                data[pair] = scores
            else:
                print(f"Warning: {pair} data length mismatch. Skipping.")
        else:
            print(f"Warning: {filename} not found.")
    return data


def create_combined_plot(x_axis, data, atom, out_file):
    """Generates the single overlay plot."""
    plt.figure(figsize=(12, 8))

    for pair in PAIR_TYPES:
        if pair in data:
            plt.plot(x_axis, data[pair], label=pair, marker=".", linewidth=1.5)

    plt.title(f"RNA Statistical Potentials (All Pairs) - {atom}", fontsize=14)
    plt.xlabel("Distance (Å)")
    plt.ylabel("Pseudo-Energy Score")
    plt.axhline(0, color="black", linestyle="-", linewidth=1)
    plt.ylim(-3, 11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Base Pair")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(out_file, dpi=300)
    print(f"Saved combined plot to: {out_file}")
    plt.close()


def create_grid_plot(x_axis, data, atom, max_dist, out_file):
    """Generates the 2x5 grid plot."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, pair in enumerate(PAIR_TYPES):
        ax = axes[i]

        if pair in data:
            ax.plot(x_axis, data[pair], marker=".", linewidth=1.5, color="tab:blue")
            ax.set_title(f"{pair} Interaction", fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", transform=ax.transAxes)
            ax.set_title(f"{pair} (Missing)", fontsize=10, color="red")

        # Styling
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_ylim(-3, 11)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Labels only on outer edges to save space
        if i >= 5:
            ax.set_xlabel("Dist (Å)")
        if i % 5 == 0:
            ax.set_ylabel("Score")

    plt.suptitle(
        f"RNA Statistical Potentials ({atom} atom) - {max_dist}Å Cutoff", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved grid plot to: {out_file}")
    plt.close()


def main():
    args = parse_arguments()

    # 1. Load Config
    try:
        atom, max_dist, bin_width = load_params(args.in_dir)
    except Exception as e:
        sys.exit(f"Error loading params: {e}")

    print(f"Generating plots for {atom} atoms...")

    # 2. Setup Axis
    nbins = int(math.ceil(max_dist / bin_width))
    x_axis = [i * bin_width + (bin_width / 2) for i in range(nbins)]

    # 3. Load Data
    data = load_all_scores(args.in_dir, len(x_axis))

    # 4. Generate Plots
    create_combined_plot(x_axis, data, atom, args.out_combined)
    create_grid_plot(x_axis, data, atom, max_dist, args.out_grid)


if __name__ == "__main__":
    main()
