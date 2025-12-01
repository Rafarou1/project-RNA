# project-RNA

**Course:** M2 GENIOMHE - RNA Structure Bioinformatics  
**Goal:** Creation of an objective function for the RNA folding problem

## Project Overview
This project implements a distance-dependent statistical potential to distinguish native RNA folds from non-native ones. The scoring function is derived from the observed frequency of distances between C3' atoms in experimentally determined structures (PDB).

The pipeline consists of three modules:
1.  **Training:** Deriving statistical potentials from PDB data.
2.  **Visualization:** Plotting interaction profiles.
3.  **Scoring:** Estimating the Gibbs free energy of new structures.

---

## Project Structure

```text
project-RNA/
├── data/
│   ├── pdb_training/       # Directory containing .pdb files
│   └── potentials/         # Output directory for generated potential files
├── train_potential.py      # Script 1: Extracts statistics and generates potentials
├── plot_potentials.py      # Script 2: Visualizes the scoring profiles
├── score_structure.py      # Script 3: Evaluates a PDB structure using the potentials
└── README.md
```

### 1\. Training

Calculates distance distributions for 10 base pairs (AA, AU, etc.) and a reference state (XX).

  * **Atom:** C3' (customizable).
  * **Constraints:** Intrachain distances only; sequence separation $|i-j| \ge 4$.
  * **Output:** Generates `potential_{pair}.txt` files and a `params.txt` configuration file.

<!-- end list -->

```bash
# Standard run with default parameters (C3', 20Å cut-off, 1Å bins)
# All parameters are customizable
python train_potential.py --pdb_dir data/pdb_training --out_dir data/potentials --verbose
```

### 2\. Visualization

Plots the pseudo-energy score as a function of distance. Requires the `params.txt` generated in step 1 to correctly scale the X-axis.

```bash
python plot_potentials.py --in_dir data/potentials --out_png potentials.png
```

### 3\. Scoring

Evaluates a target structure by summing the pseudo-energy of all valid pairwise interactions using linear interpolation.

```bash
python score_structure.py --pdb data/pdb_training/1EHZ.pdb --pot_dir data/potentials
```

-----

## Current Status & Preliminary Results

**Dataset:**
We have currently established the pipeline using a **single PDB file (`1EHZ.pdb`)** as a proof-of-concept.

**Observations:**

  * **Potentials:** The resulting plots (`potentials.png`) are "jagged" with many values hitting the maximum penalty ceiling (+10.0). This is expected because a single structure can't provide data for all distance bins for all base pairs.
  * **Scoring:** The native structure `1EHZ` currently yields a positive pseudo-energy (here `+14.63`). Since statistical potentials should yield negative scores for native folds, this confirms that the current training set is too sparse.

**Next Steps:**

  * [ ] **Scale Up:** Run `train_potential.py` on a large dataset (RNA-Puzzles or non-redundant PDB list) to smooth the interaction curves.
  * [ ] **KDE:** Implement Kernel Density Estimation to replace discrete histograms for better continuity (if possible in Python).

-----

## Requirements

  * Python 3.8+
  * Matplotlib

<!-- end list -->
