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
from rna_utils import parse_pdb_atoms, pair_key, load_params, PAIR_TYPES, VALID_BASES


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Score an RNA structure with a distance-dependent potential."
    )
    parser.add_argument("--pdb", type=str, required=True, help="PDB file to score.")
    parser.add_argument(
        "--pot_dir",
        type=str,
        required=True,
        help="Directory containing potential_XX.txt files.",
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


def main():
    args = parse_arguments()

    # Load Training Parameters
    atom_type, max_dist, bin_width = load_params(args.pot_dir)
    nbins = int(math.ceil(max_dist / bin_width))

    if args.verbose:
        print(f"Loaded Params: Atom={atom_type}, MaxDist={max_dist}, Width={bin_width}")

    # Load Potentials
    potentials = load_potentials(args.pot_dir, nbins)
    if not potentials:
        sys.exit("No potentials loaded.")

    # Parse PDB
    chains = parse_pdb_atoms(args.pdb, atom_type)
    if not chains:
        print(f"No valid {atom_type} atoms found in {args.pdb}.")
        print("Estimated pseudo-energy: 0.0")
        return

    # Calculate Score
    total_score = 0.0
    pairs_used = 0

    for chain_id, residues in chains.items():
        n = len(residues)
        for i in range(n):
            for j in range(i + 4, n):
                r1_name = residues[i][1]
                r1_coords = residues[i][2]

                r2_name = residues[j][1]
                r2_coords = residues[j][2]

                d = math.dist(r1_coords, r2_coords)

                if d < max_dist:
                    key = pair_key(r1_name, r2_name)

                    if key in potentials:
                        score = interpolate_score(
                            d, potentials[key], bin_width, max_dist
                        )
                        total_score += score
                        pairs_used += 1

    print(f"Structure: {os.path.basename(args.pdb)}")
    print(f"Number of pairs used: {pairs_used}")
    print(f"Estimated pseudo-energy: {total_score:.4f}")


if __name__ == "__main__":
    main()
