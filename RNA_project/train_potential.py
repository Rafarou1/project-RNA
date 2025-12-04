#!/usr/bin/env python3

import os
import math
import argparse
import sys

try:
    from rna_utils import parse_pdb_atoms, get_bin_index, pair_key, PAIR_TYPES
except ImportError:
    sys.exit("Error: rna_utils.py must be in the same directory.")


class RNAPotentialTrainer:

    def __init__(self, atom_type="C3'", max_dist=20.0, bin_width=1.0):
        self.atom_type = atom_type
        self.max_dist = max_dist
        self.bin_width = bin_width
        
        self.nbins = int(math.ceil(self.max_dist / self.bin_width))
        

        # pair_counts: Stores N_ij(r)
        # ref_counts: Stores N_xx(r)
        self.pair_counts = {p: [0] * self.nbins for p in PAIR_TYPES}
        self.ref_counts = [0] * self.nbins
        
        self.files_processed = 0

    def train(self, pdb_dir, verbose=False):
        files = [
            os.path.join(pdb_dir, f) 
            for f in os.listdir(pdb_dir) 
            if f.endswith(".pdb")
        ]

        if not files:
            sys.exit(f"No PDB files found in {pdb_dir}")

        print(f"Starting training on {len(files)} files...")

        for fpath in files:
            if verbose:
                print(f"Processing: {os.path.basename(fpath)}")
            
            try:
                self._process_single_file(fpath)
                self.files_processed += 1
            except Exception as e:
                print(f"Warning: Failed to process {fpath}: {e}")

    def _process_single_file(self, fpath):
        chains = parse_pdb_atoms(fpath, self.atom_type)

        for residues in chains.values():
            n_res = len(residues)
            # loop j starts only at i + 4.
            for i in range(n_res):
                for j in range(i + 4, n_res):
                    self._add_observation(residues[i], residues[j])

    def _add_observation(self, res_i, res_j):
        # residues[x]: (res_num, res_name, (x, y, z))
        coords_i = res_i[2]
        coords_j = res_j[2]
        
        dist = math.dist(coords_i, coords_j)

        idx = get_bin_index(dist, self.max_dist, self.bin_width)

        if idx is not None:
            # Determine nucleotide pair types
            key = pair_key(res_i[1], res_j[1])

            if key in self.pair_counts:
                self.pair_counts[key][idx] += 1
                self.ref_counts[idx] += 1

    def save_potentials(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print("Calculating potentials and writing files...")

        EPS = 1e-12
        
        total_ref = sum(self.ref_counts) + EPS

        for pair in PAIR_TYPES:
            scores = []
            total_pair = sum(self.pair_counts[pair]) + EPS

            for r in range(self.nbins):
                freq_obs = self.pair_counts[pair][r] / total_pair
                
                freq_ref = self.ref_counts[r] / total_ref

                if freq_obs < EPS:
                    score = 10.0 # Arbitrary high penalty for unobserved, safety measure tbh
                else:
                    ratio = freq_obs / (freq_ref + EPS)
                    if ratio > 0:
                        score = -math.log(ratio)
                    else:
                        score = 10.0

                score = min(max(score, -10.0), 10.0)
                scores.append(score)

            self._write_profile(out_dir, pair, scores)

        self._write_metadata(out_dir)
        print(f"Results saved to {out_dir}")

    def _write_profile(self, out_dir, pair, scores):
        """Writes the 20 lines of scores for a specific pair."""
        filename = f"potential_{pair}.txt"
        path = os.path.join(out_dir, filename)
        with open(path, "w") as f:
            for s in scores:
                f.write(f"{s:.6f}\n")
        print(f"Wrote {path}")

    def _write_metadata(self, out_dir):
        """Writes parameter files required for the scoring script."""
        # This is useful for the subsequent scoring script to know the params
        params_path = os.path.join(out_dir, "params.txt")
        with open(params_path, "w") as f:
            f.write(f"{self.atom_type}\n")
            f.write(f"{self.max_dist}\n")
            f.write(f"{self.bin_width}\n")


def parse_arguments():
    """
    Parses command line arguments compatible with the pipeline runner.
    """
    parser = argparse.ArgumentParser(
        description="Train RNA potential (C3', i+4) as per TP_RNA instructions."
    )
    
    # Required path arguments, basic
    parser.add_argument(
        "--pdb_dir", type=str, required=True, 
        help="Input directory of PDBs"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, 
        help="Output directory for potentials"
    )

    # Optional configuration arguments; required by pipeline runner and the GUI wrapper we made
    parser.add_argument(
        "--atom", type=str, default="C3'", 
        help="Atom type to use (default: C3')"
    )
    parser.add_argument(
        "--max_dist", type=float, default=20.0, 
        help="Max distance in Angstroms (default: 20.0)"
    )
    parser.add_argument(
        "--bin_width", type=float, default=1.0, 
        help="Bin width in Angstroms (default: 1.0)"
    )
    
    parser.add_argument(
        "--verbose", action="store_true", 
        help="Print detailed progress"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Initialize the trainer with the arguments from the pipeline
    trainer = RNAPotentialTrainer(
        atom_type=args.atom,
        max_dist=args.max_dist,
        bin_width=args.bin_width
    )
    
    trainer.train(args.pdb_dir, verbose=args.verbose)
    trainer.save_potentials(args.out_dir)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()