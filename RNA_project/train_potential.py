#!/usr/bin/env python3
"""
train_potential.py

Train an RNA distance-dependent statistical potential using C3' atoms.

Usage:
    python train_potential.py --pdb_dir data/pdb_training --out_dir data/potentials

Details:
- Only C3' atoms
- Only intrachain distances
- Only residues with sequence separation >= 4 (i, i+4, i+5, ...)
- Distances in [0, 20] Å, 20 bins of width 1 Å
- 10 base-pair types: AA, AU, AC, AG, UU, UC, UG, CC, CG, GG
- Reference distribution "XX" (all residue pairs pooled)
"""

import os
import math
import argparse
from collections import defaultdict, OrderedDict

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
        description="Train RNA distance-dependent potential from PDB structures."
    )
    parser.add_argument(
        "--pdb_dir", type=str, required=True,
        help="Directory containing PDB files for training."
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for potential profiles."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress and summary information during training."
    )
    return parser.parse_args()


def list_pdb_files(pdb_dir):
    files = []
    for name in os.listdir(pdb_dir):
        if name.lower().endswith(".pdb"):
            files.append(os.path.join(pdb_dir, name))
    files.sort()
    return files


def parse_c3_atoms(pdb_path):
    """
    Parse C3' atoms from a PDB file.

    Returns a dict:
        chain_id -> list of (seq_index_in_chain, resname, (x,y,z))
    seq_index_in_chain is simply the order in which residues appear per chain.
    """
    chains = {}
    # To avoid multiple entries per residue, we track last residue seen in each chain
    last_resid = {}

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain_id = line[21].strip() or " "
            resseq = line[22:26].strip()
            # altLoc (column 17 in PDB, index 16)
            # we ignore altLoc if not space or 'A'
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

            # Only add one C3' per residue
            if last_resid[chain_id] != resid:
                chains[chain_id].append((len(chains[chain_id]), resname, (x, y, z)))
                last_resid[chain_id] = resid

    return chains


def distance(coord1, coord2):
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    dz = coord1[2] - coord2[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def get_bin_index(d):
    if d < 0.0 or d > MAX_DIST:
        return None
    # Put d == MAX_DIST in the last bin
    idx = int(d // BIN_WIDTH)
    if idx >= NBINS:
        idx = NBINS - 1
    return idx


def pair_key(res1, res2):
    """
    Return canonical pair type from two bases (unordered).
    e.g. A/U -> AU, U/A -> AU, C/G -> CG, etc.
    """
    pair = "".join(sorted([res1, res2]))
    return pair


def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    pdb_files = list_pdb_files(args.pdb_dir)
    if not pdb_files:
        raise SystemExit(f"No PDB files found in {args.pdb_dir}")

    verbose = bool(getattr(args, "verbose", False))

    # Initialize counts
    pair_bin_counts = {p: [0] * NBINS for p in PAIR_TYPES}
    ref_bin_counts = [0] * NBINS  # XX reference

    pair_total_counts = {p: 0 for p in PAIR_TYPES}
    ref_total_count = 0

    # --- Collect statistics from all PDBs ---
    processed_pdbs = 0
    for pdb in pdb_files:
        processed_pdbs += 1
        if verbose:
            # report estimated residues parsed in file
            chains_tmp = parse_c3_atoms(pdb)
            nres = sum(len(v) for v in chains_tmp.values())
            print(f"Processing {os.path.basename(pdb)} ({processed_pdbs}/{len(pdb_files)}): chains={len(chains_tmp)}, residues_with_C3'={nres}")
        chains = parse_c3_atoms(pdb)
        chains = parse_c3_atoms(pdb)
        for chain_id, residues in chains.items():
            n = len(residues)
            for i in range(n):
                idx_i, res_i, coord_i = residues[i]
                for j in range(i + 4, n):  # i, i+4, i+5, ...
                    idx_j, res_j, coord_j = residues[j]
                    d = distance(coord_i, coord_j)
                    if d > MAX_DIST:
                        continue
                    bin_idx = get_bin_index(d)
                    if bin_idx is None:
                        continue

                    # Reference "XX" all pairs
                    ref_bin_counts[bin_idx] += 1
                    ref_total_count += 1

                    # Specific base pair
                    pk = pair_key(res_i, res_j)
                    if pk in pair_bin_counts:
                        pair_bin_counts[pk][bin_idx] += 1
                        pair_total_counts[pk] += 1

    if ref_total_count == 0:
        raise SystemExit("No distances collected. Check input PDB files.")

    # --- Compute reference frequencies f_REF(XX) ---
    f_ref = [c / ref_total_count for c in ref_bin_counts]

    # Avoid exact zeros in reference: tiny pseudocount
    EPS = 1e-12
    f_ref = [max(fr, EPS) for fr in f_ref]

    # --- Compute potentials for each pair type ---
    for pair in PAIR_TYPES:
        counts = pair_bin_counts[pair]
        total = pair_total_counts[pair]

        if total == 0:
            # No data for this pair: set all scores to max penalty 10
            scores = [10.0] * NBINS
        else:
            f_obs = [c / total for c in counts]
            # Avoid zeros in observation frequencies
            f_obs = [max(fo, EPS) for fo in f_obs]

            scores = []
            for fr_obs, fr_ref in zip(f_obs, f_ref):
                ratio = fr_obs / fr_ref
                # If ratio <= 0 (should not happen with EPS), set max score
                if ratio <= 0:
                    score = 10.0
                else:
                    score = -math.log(ratio)
                    if score > 10.0:
                        score = 10.0
                scores.append(score)

        out_path = os.path.join(args.out_dir, f"potential_{pair}.txt")
        with open(out_path, "w") as out_f:
            for s in scores:
                out_f.write(f"{s:.6f}\n")

        print(f"Wrote {out_path}")

    # write a short summary to out_dir
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as s_f:
        s_f.write(f"pdb_files_processed: {processed_pdbs}\n")
        s_f.write(f"total_distance_counts: {ref_total_count}\n")
        s_f.write("counts_per_pair:\n")
        for p in PAIR_TYPES:
            s_f.write(f"  {p}: {pair_total_counts[p]}\n")

    if verbose:
        print(f"Wrote summary to {summary_path}")
        print("Training completed.")
    else:
        print("Training completed.")


if __name__ == "__main__":
    main()
