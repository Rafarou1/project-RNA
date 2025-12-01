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
import sys
import math
from rna_utils import load_params, PAIR_TYPES


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot RNA distance-dependent potentials."
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Directory containing potential_XX.txt files.",
    )
    parser.add_argument(
        "--out_png", type=str, required=True, help="Path to output PNG figure."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load parameters
    atom, max_dist, bin_width = load_params(args.in_dir)
    print(f"Plotting for {atom} atoms (Max: {max_dist}A, Width: {bin_width}A)")

    # Set-up for plotting
    nbins = int(math.ceil(max_dist / bin_width))
    x_axis = [i * bin_width + (bin_width / 2) for i in range(nbins)]

    plt.figure(figsize=(12, 8))

    # Plot each pair type
    for pair in PAIR_TYPES:
        fname = os.path.join(args.in_dir, f"potential_{pair}.txt")
        if os.path.exists(fname):
            with open(fname, "r") as f:
                scores = [float(line.strip()) for line in f if line.strip()]
            if len(scores) == len(x_axis):
                plt.plot(x_axis, scores, label=pair, marker=".", linewidth=1)
            else:
                print(
                    f"Skipping {pair}: Data length {len(scores)} != Bins {len(x_axis)}"
                )
        else:
            print(f"Warning: {fname} not found.")

    # Formatting the plot
    plt.xlabel("Distance (Å)")
    plt.ylabel("Pseudo-energy score")
    plt.title(f"RNA statistical potentials on {atom} atoms")
    plt.axhline(0, color="black", linestyle="-", linewidth=1)
    plt.ylim(-11, 11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(args.out_png, dpi=300)
    print(f"Saved figure to {args.out_png}")


if __name__ == "__main__":
    main()
