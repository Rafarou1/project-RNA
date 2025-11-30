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

PAIR_TYPES = [
    "AA", "AU", "AC", "AG",
    "UU", "UC", "UG",
    "CC", "CG",
    "GG",
]

BIN_WIDTH = 1.0
MAX_DIST = 20.0
NBINS = int(MAX_DIST / BIN_WIDTH)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot RNA distance-dependent potentials."
    )
    parser.add_argument(
        "--in_dir", type=str, required=True,
        help="Directory containing potential_XX.txt files."
    )
    parser.add_argument(
        "--out_png", type=str, required=True,
        help="Path to output PNG figure."
    )
    return parser.parse_args()


def load_profile(path):
    with open(path, "r") as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return values


def main():
    args = parse_arguments()

    distances = [i * BIN_WIDTH + BIN_WIDTH / 2.0 for i in range(NBINS)]

    plt.figure(figsize=(10, 6))

    for pair in PAIR_TYPES:
        fname = os.path.join(args.in_dir, f"potential_{pair}.txt")
        if not os.path.exists(fname):
            print(f"Warning: missing {fname}, skipping {pair}")
            continue
        scores = load_profile(fname)
        if len(scores) != NBINS:
            print(f"Warning: {fname} does not have {NBINS} lines, skipping")
            continue
        plt.plot(distances, scores, label=pair)

    plt.xlabel("C3'-C3' distance (Å)")
    plt.ylabel("Pseudo-energy score")
    plt.title("RNA distance-dependent potentials")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"Saved figure to {args.out_png}")
    # Uncomment if you want an interactive window:
    # plt.show()


if __name__ == "__main__":
    main()
