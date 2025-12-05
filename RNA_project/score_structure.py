#!/usr/bin/env python3
"""
score_structure.py

Compute the estimated Gibbs free energy (pseudo-energy) of an RNA structure
using pre-trained C3'-C3' distance-dependent potentials.

Usage:
    python score_structure.py --pdb data/pdb_training/1EHZ.pdb --pot_dir data/potentials
"""

import os
import math
import argparse
import sys
from rna_utils import (
    parse_pdb_atoms,
    pair_key,
    load_params,
    iterate_valid_pairs,
    PAIR_TYPES,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Score an RNA structure with a distance-dependent potential."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdb", help="Single PDB file to score.")
    group.add_argument("--pdb_dir", help="Directory containing PDB files to score.")

    parser.add_argument(
        "--pot_dir",
        required=True,
        help="Directory containing potentials and params.txt.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed info.")
    return parser.parse_args()


def load_potentials(pot_dir, nbins):
    potentials = {}
    for pair in PAIR_TYPES:
        fname = os.path.join(pot_dir, f"potential_{pair}.txt")
        if not os.path.exists(fname):
            print(f"Warning: Missing potential file {fname}")
            continue
        with open(fname, "r") as f:
            scores = [float(line.strip()) for line in f if line.strip()]
        if len(scores) != nbins:
            print(f"Warning: {pair} has {len(scores)} bins, expected {nbins}.")

        potentials[pair] = scores
    return potentials


def interpolate_score(dist, scores, bin_width, max_dist):
    """
    Linear interpolation of score for distance d given a discrete profile 'scores'.

    bins: 0-1, 1-2, ..., 19-20 Å (NBINS = 20)
    We treat score[i] as the value at the center of bin i, i.e., (i+0.5)*BIN_WIDTH.

    A simple approach:
    - if d <= 0: use first score
    - if d >= MAX_DIST: use last score
    - else: find surrounding bin centers and interpolate.
    """
    if dist <= 0.0:
        return scores[0]
    if dist >= max_dist:
        return scores[-1]

    # position in bins
    # center of bin i is (i+0.5)*BIN_WIDTH
    val_idx = dist / bin_width - 0.5
    idx_low = int(math.floor(val_idx))
    idx_high = idx_low + 1
    if idx_low <= 0:
        return scores[0]
    if idx_high >= len(scores):
        return scores[-1]

    frac = val_idx - idx_low

    score_low = scores[idx_low]
    score_high = scores[idx_high]

    # Linear interpolation
    return score_low + (score_high - score_low) * frac


def score_single_file(pdb_path, atom_type, potentials, bin_width, max_dist):
    """Helper function to score one file."""
    chains = parse_pdb_atoms(pdb_path, atom_type)
    if not chains:
        return None, 0

    total_score = 0.0
    pairs_used = 0

    for r1_name, r1_coords, r2_name, r2_coords in iterate_valid_pairs(chains):
        d = math.dist(r1_coords, r2_coords)

        if d < max_dist:
            key = pair_key(r1_name, r2_name)
            if key in potentials:
                score = interpolate_score(d, potentials[key], bin_width, max_dist)
                total_score += score
                pairs_used += 1

    return total_score, pairs_used


def main():
    args = parse_arguments()

    # Load Training Parameters
    atom_type, max_dist, bin_width = load_params(args.pot_dir)
    nbins = int(math.ceil(max_dist / bin_width))

    # Load Potentials
    potentials = load_potentials(args.pot_dir, nbins)
    if not potentials:
        sys.exit("No potentials loaded.")

    files_to_score = []
    if args.pdb:
        files_to_score.append(args.pdb)
    elif args.pdb_dir:
        if not os.path.exists(args.pdb_dir):
            sys.exit(f"Error: Directory {args.pdb_dir} not found.")
        files_to_score = [
            os.path.join(args.pdb_dir, f)
            for f in os.listdir(args.pdb_dir)
            if f.endswith(".pdb")
        ]

    if not files_to_score:
        print("No PDB files found to score.")
        return

    print(f"Scoring {len(files_to_score)} structure(s) using atom {atom_type}...")
    print(f"{'Structure':<30} {'Pairs':<10} {'Energy':<15}")
    print("-" * 60)

    # 3. Loop and Score
    for pdb_file in files_to_score:
        score, count = score_single_file(
            pdb_file, atom_type, potentials, bin_width, max_dist
        )
        name = os.path.basename(pdb_file)

        if score is None:
            print(f"{name:<30} {'0':<10} {'ERROR':<15}")
        else:
            print(f"{name:<30} {count:<10} {score:<15.4f}")


if __name__ == "__main__":
    main()
