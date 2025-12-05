#!/usr/bin/env python3
"""
train_potential.py

Train an RNA distance-dependent statistical potential using C3' atoms.

Usage:
    python train_potential.py --pdb_dir data/pdb_training --out_dir data/potentials

Details:
- Can change but focusing on C3' atoms
- Only intrachain distances
- Only residues with sequence separation >= 4 (i, i+4, i+5, ...)
- Distances in [0, 20] Å, 20 bins of width 1 Å
- 10 base-pair types: AA, AU, AC, AG, UU, UC, UG, CC, CG, GG
- Reference distribution "XX" (all residue pairs pooled)
"""

import os
import math
import argparse
import sys
from rna_utils import parse_pdb_atoms, get_bin_index, pair_key, PAIR_TYPES


def parse_arguments():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Train RNA distance-dependent potential from PDB structures."
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        required=True,
        help="Directory containing PDB files for training.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for potential profiles.",
    )
    parser.add_argument("--atom", default="C3'", help="Atom type to use (default: C3')")
    parser.add_argument(
        "--max_dist", type=float, default=20.0, help="Max distance in Å (default: 20.0)"
    )
    parser.add_argument(
        "--bin_width", type=float, default=1.0, help="Bin width in Å (default: 1.0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and summary information during training.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    nbins = int(math.ceil(args.max_dist / args.bin_width))
    # Initialize counts
    pair_counts = {p: [0] * nbins for p in PAIR_TYPES}
    ref_counts = [0] * nbins

    files = [
        os.path.join(args.pdb_dir, f)
        for f in os.listdir(args.pdb_dir)
        if f.endswith(".pdb")
    ]
    if not files:
        sys.exit(f"No PDB files found in {args.pdb_dir}")

    print(f"Starting training on {len(files)} PDB files...")
    print(
        f"Configuration: Atom={args.atom}, MaxDist={args.max_dist}A, BinWidth={args.bin_width}A"
    )

    # 1. Collect statistics from all PDBs
    processed_count = 0
    for fpath in files:
        if args.verbose:
            print(f"Processing: {os.path.basename(fpath)}")
        chains = parse_pdb_atoms(fpath, args.atom)
        for chain_id, residues in chains.items():
            n = len(residues)
            # Consider all residue pairs with separation >= 4
            for i in range(n):
                for j in range(i + 4, n):  # i, i+4, i+5, ...
                    # Extract coordinates + compute distance
                    coords_i = residues[i][2]
                    coords_j = residues[j][2]
                    d = math.dist(coords_i, coords_j)

                    # Determine bin index
                    idx = get_bin_index(d, args.max_dist, args.bin_width)

                    if idx is not None:
                        # Determine pair type
                        res_i = residues[i][1]
                        res_j = residues[j][1]
                        key = pair_key(res_i, res_j)

                        if key in pair_counts:
                            pair_counts[key][idx] += 1
                            ref_counts[idx] += 1
        processed_count += 1

    print("Calculating potentials...")

    # Avoid exact zeros in reference: tiny pseudocount
    EPS = 1e-12
    total_ref = sum(ref_counts) + EPS

    # 2. Compute potentials for each pair type
    for pair in PAIR_TYPES:
        # A. Save Raw Counts
        count_file = os.path.join(args.out_dir, f"counts_{pair}.txt")
        with open(count_file, "w") as f:
            for c in pair_counts[pair]:
                f.write(f"{c}\n")

        # B. Compute Scores & Calculate Potential
        scores = []
        total_pair = sum(pair_counts[pair]) + EPS

        for i in range(nbins):
            # Probability observed and reference
            p_obs = pair_counts[pair][i] / total_pair
            p_ref = ref_counts[i] / total_ref

            if p_obs < EPS:
                # If we never observed this distance, assign high energy penalty
                score = 10.0
            else:
                ratio = p_obs / (p_ref + EPS)
                if ratio > 0:
                    score = -math.log(ratio)
                else:
                    score = 10.0
                # Clamp scores to avoid extreme values
                score = min(max(score, -10.0), 10.0)

            scores.append(score)

        out_path = os.path.join(args.out_dir, f"potential_{pair}.txt")
        with open(out_path, "w") as out_f:
            for s in scores:
                out_f.write(f"{s:.6f}\n")

        print(f"Wrote {out_path}")

    # 3. Saving data
    with open(os.path.join(args.out_dir, "counts_XX.txt"), "w") as f:
        for c in ref_counts:
            f.write(f"{c}\n")

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as s_f:
        s_f.write(f"pdb_files_processed: {processed_count}\n")
        s_f.write(f"total_distance_counts: {sum(ref_counts)}\n")
        s_f.write(
            f"parameters used: Atom={args.atom}, Max={args.max_dist}, Width={args.bin_width}\n"
        )
        s_f.write("counts_per_pair:\n")
        for p in PAIR_TYPES:
            # Use pair_counts (aggregated total), not bins directly
            total_for_pair = sum(pair_counts[p])
            s_f.write(f"  {p}: {total_for_pair}\n")

    params_path = os.path.join(args.out_dir, "params.txt")
    with open(params_path, "w") as p_f:
        p_f.write(f"{args.atom}\n")
        p_f.write(f"{args.max_dist}\n")
        p_f.write(f"{args.bin_width}\n")

    print(f"Training complete. Summary saved to {summary_path}")
    print(f"Parameters saved to {params_path}")

    print(f"Training complete. Processed {processed_count} files.")
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
