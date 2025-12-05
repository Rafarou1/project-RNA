#!/usr/bin/env python3
import os
import math
import argparse
import sys
# This is a custom helper module. It handles the messy parts of parsing PDBs 
# and managing the specific file formats we are using.
from rna_utils import (
    parse_pdb_atoms,
    pair_key,
    load_params,
    load_pair_data,
    iterate_valid_pairs,
)


def parse_arguments():
    """
    Sets up the command line interface so you can run the script easily from a terminal.
    """
    parser = argparse.ArgumentParser(
        description="Score an RNA structure with a distance-dependent potential."
    )
    # We force the user to choose: score one specific file OR a whole folder of them.
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


def interpolate_score(dist, scores, bin_width, max_dist):
    """
    Calculates the score for a specific distance using linear interpolation.

    The 'scores' list is discrete (like steps on a stair), but distance is continuous 
    (like a ramp). This function smooths the data. If we have a score for 5.0A and 6.0A, 
    but our distance is 5.5A, we take the average of the two scores.
    """
    # 1. Handle edge cases: 
    # If atoms are impossibly close or too far apart, we just clamp to the 
    # first or last known score rather than crashing.
    if dist <= 0.0:
        return scores[0]
    if dist >= max_dist:
        return scores[-1]

    # 2. Find where we sit in the list of bins.
    # We treat the value at index 'i' as the centre of that bin.
    val_idx = dist / bin_width - 0.5
    idx_low = int(math.floor(val_idx))
    idx_high = idx_low + 1

    # Safety check: ensure our indices don't fall off the edge of the list.
    if idx_low <= 0:
        return scores[0]
    if idx_high >= len(scores):
        return scores[-1]

    # 3. Calculate the weighted average (Linear Interpolation).
    # 'frac' tells us how close we are to the higher bin.
    frac = val_idx - idx_low

    score_low = scores[idx_low]
    score_high = scores[idx_high]

    return score_low + (score_high - score_low) * frac


def score_single_file(pdb_path, atom_type, potentials, bin_width, max_dist):
    """
    The main calculation engine for a single RNA file. 
    It iterates through atom pairs, measures distance, and sums up the energy.
    """
    # Parse the PDB file to get a clean list of atom coordinates.
    chains = parse_pdb_atoms(pdb_path, atom_type)
    if not chains:
        return None, 0

    total_score = 0.0
    pairs_used = 0

    # We loop through every valid combination of two nucleotides.
    # 'iterate_valid_pairs' (from utils) ensures we don't score neighbours 
    # (i and i+1) which are covalently bonded and skew the results.
    for r1_name, r1_coords, r2_name, r2_coords in iterate_valid_pairs(chains):
        d = math.dist(r1_coords, r2_coords)

        # We only care about pairs within our defined cutoff (max_dist).
        # Anything further away is considered negligible interaction.
        if d < max_dist:
            # Create a lookup key (e.g., "A-U") to find the right potential profile.
            key = pair_key(r1_name, r2_name)
            if key in potentials:
                # Get the precise score for this distance and add it to the total.
                score = interpolate_score(d, potentials[key], bin_width, max_dist)
                total_score += score
                pairs_used += 1

    return total_score, pairs_used


def main():
    args = parse_arguments()

    # 1. Load the 'Rules' (Training Parameters and Potentials)
    # These files tell us what 'good' and 'bad' distances look like based on known structures.
    atom_type, max_dist, bin_width = load_params(args.pot_dir)
    nbins = int(math.ceil(max_dist / bin_width))

    potentials = load_pair_data(args.pot_dir, nbins, prefix="potential")
    if not potentials:
        sys.exit("No potentials loaded.")

    # 2. Build the list of files to process (either one file or a whole folder)
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

    # Print a tidy header for the output table
    print(f"Scoring {len(files_to_score)} structure(s) using atom {atom_type}...")
    print(f"{'Structure':<30} {'Pairs':<10} {'Energy':<15}")
    print("-" * 60)

    # 3. Loop and Score
    # Process each file one by one and print the result immediately.
    for pdb_file in files_to_score:
        score, count = score_single_file(
            pdb_file, atom_type, potentials, bin_width, max_dist
        )
        name = os.path.basename(pdb_file)

        if score is None:
            print(f"{name:<30} {'0':<10} {'ERROR':<15}")
        else:
            # Print formatted columns: Name, Number of pairs used, Final Energy Score.
            print(f"{name:<30} {count:<10} {score:<15.4f}")


if __name__ == "__main__":
    main()