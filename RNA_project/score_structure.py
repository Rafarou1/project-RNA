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

BIN_WIDTH = 1.0
MAX_DIST = 20.0
NBINS = int(MAX_DIST / BIN_WIDTH)

VALID_BASES = {"A", "U", "C", "G"}

PAIR_TYPES = [
    "AA", "AU", "AC", "AG",
    "UU", "UC", "UG",
    "CC", "CG",
    "GG",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Score an RNA structure with a distance-dependent potential."
    )
    parser.add_argument(
        "--pdb", type=str, required=True,
        help="PDB file to score."
    )
    parser.add_argument(
        "--pot_dir", type=str, required=True,
        help="Directory containing potential_XX.txt files."
    )
    return parser.parse_args()


def load_potentials(pot_dir):
    profiles = {}
    for pair in PAIR_TYPES:
        fname = os.path.join(pot_dir, f"potential_{pair}.txt")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing potential file: {fname}")
        with open(fname, "r") as f:
            vals = [float(line.strip()) for line in f if line.strip()]
        if len(vals) != NBINS:
            raise ValueError(f"{fname} must have {NBINS} lines.")
        profiles[pair] = vals
    return profiles


def parse_c3_atoms(pdb_path):
    """
    Same as in training: returns chain_id -> list of (seq_index, resname, (x,y,z))
    """
    chains = {}
    last_resid = {}

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain_id = line[21].strip() or " "
            resseq = line[22:26].strip()
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue

            if atom_name != "C3'":
                continue
            if resname not in VALID_BASES:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            resid = (chain_id, resseq)
            if chain_id not in chains:
                chains[chain_id] = []
                last_resid[chain_id] = None

            if last_resid[chain_id] != resid:
                chains[chain_id].append((len(chains[chain_id]), resname, (x, y, z)))
                last_resid[chain_id] = resid

    return chains


def distance(coord1, coord2):
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    dz = coord1[2] - coord2[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def pair_key(res1, res2):
    return "".join(sorted([res1, res2]))


def interpolate_score(d, scores):
    """
    Linear interpolation of score for distance d given a discrete profile 'scores'.

    bins: 0-1, 1-2, ..., 19-20 Å (NBINS = 20)
    We treat score[i] as the value at the center of bin i, i.e., (i+0.5)*BIN_WIDTH.

    A simple approach:
    - if d <= 0: use first score
    - if d >= MAX_DIST: use last score
    - else: find surrounding bin centers and interpolate.
    """
    if d <= 0.0:
        return scores[0]
    if d >= MAX_DIST:
        return scores[-1]

    # position in bins
    # center of bin i is (i+0.5)*BIN_WIDTH
    x = d / BIN_WIDTH - 0.5
    if x <= 0:
        return scores[0]
    if x >= NBINS - 1:
        return scores[-1]

    i_low = int(math.floor(x))
    i_high = i_low + 1
    frac = x - i_low

    s_low = scores[i_low]
    s_high = scores[i_high]

    return (1.0 - frac) * s_low + frac * s_high


def main():
    args = parse_arguments()

    potentials = load_potentials(args.pot_dir)
    chains = parse_c3_atoms(args.pdb)

    total_score = 0.0
    n_pairs_used = 0

    for chain_id, residues in chains.items():
        n = len(residues)
        for i in range(n):
            idx_i, res_i, coord_i = residues[i]
            for j in range(i + 4, n):
                idx_j, res_j, coord_j = residues[j]
                d = distance(coord_i, coord_j)
                if d > MAX_DIST:
                    continue

                pk = pair_key(res_i, res_j)
                if pk not in potentials:
                    continue

                score = interpolate_score(d, potentials[pk])
                total_score += score
                n_pairs_used += 1

    print(f"Structure: {args.pdb}")
    print(f"Number of C3'-C3' pairs used: {n_pairs_used}")
    print(f"Estimated pseudo-energy: {total_score:.4f}")


if __name__ == "__main__":
    main()
