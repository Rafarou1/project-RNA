import os
import sys

VALID_BASES = {"A", "U", "C", "G"}
PAIR_TYPES = [
    "AA",
    "AU",
    "AC",
    "AG",
    "UU",
    "UC",
    "UG",
    "CC",
    "CG",
    "GG",
]


def parse_pdb_atoms(pdb_path, atom_type):
    """
    Parse atoms from a PDB file.
    Returns a dict: chain_id -> list of (seq_index_in_chain, resname, (x,y,z))
    """
    chains = {}
    last_resid = {}

    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ENDMDL"):
                    break
                if not line.startswith("ATOM"):
                    continue

                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain_id = line[21].strip() or " "
                resseq = line[22:26].strip()
                icode = line[26]

                if atom_name != atom_type or resname not in VALID_BASES:
                    continue

                altloc = line[16].strip()
                if altloc not in ("", "A"):
                    continue

                resid = (chain_id, resseq, icode)
                try:
                    coords = (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                    if chain_id not in chains:
                        chains[chain_id] = []
                        last_resid[chain_id] = None
                    if last_resid[chain_id] != resid:
                        chains[chain_id].append(
                            (len(chains[chain_id]), resname, coords)
                        )
                        last_resid[chain_id] = resid
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Warning: PDB file not found: {pdb_path}")
        return {}

    return chains


def get_bin_index(dist, max_dist, width):
    """Return bin index for given distance, or None if out of range."""
    if dist >= max_dist or dist < 0:
        return None
    return int(dist // width)


def pair_key(res1, res2):
    """Return canonical pair type from two bases (unordered)."""
    return "".join(sorted([res1, res2]))


def load_params(path):
    """Load training parameters from params.txt in the specified directory."""
    params_path = os.path.join(path, "params.txt")
    try:
        with open(params_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            atom = lines[0]
            max_dist = float(lines[1])
            bin_width = float(lines[2])
            return atom, max_dist, bin_width
    except Exception as e:
        print(f"Error reading parameters from {params_path}: {e}")
        sys.exit(1)
