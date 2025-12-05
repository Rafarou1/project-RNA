# Project-RNA

**Course:** M2 GENIOMHE - RNA Structure Bioinformatics  
**Goal:** Creation of an objective function for the RNA folding problem

## Project Overview

This project implements a distance-dependent statistical potential to distinguish native RNA folds from non-native ones. The scoring function is derived from the observed frequency of distances between C3' atoms in experimentally determined structures (PDB).

The project has been evolved into a modern Streamlit Dashboard, transforming the original discrete scripts into an interactive bioinformatics pipeline while retaining full command-line utility.

## Project Structure

```text
project-RNA/RNA_project/
├── data/
│   ├── pdb_training/      # Directory for training PDB files
│   └── potentials_multi/  # Output directory for generated potentials
├── app/
│   ├── app.py             # Script for the application
├── run_pipeline.py        # Master script: Runs Training -> Plotting -> Scoring
├── train_potential.py     # Training logic: Extracts statistics from PDBs
├── plot_potentials.py     # Plotting logic: Generates Potentials & Histograms
├── score_structure.py     # Scoring logic: Evaluates structures
├── rna_utils.py           # Shared library (Parsing, Math, Loading)
├── config.py              # Centralized path configuration
└── README.md              # Here you are!
```

-----

## Usage: Interactive Web Dashboard

The web interface provides a guided workflow for training, visualising, and scoring without manually handling command-line arguments.

### **1. Launch the App**

```bash
streamlit run app.py
```

### **2. Dashboard Features**

  * **Welcome & Data Deposit:**
      * **Drag and Drop:** Supports uploading individual PDB files or dragging entire folders.
      * **In Memory Caching:** Uploaded files are processed in a persistent session state, eliminating the need for manual file system management (Stored in RAM). 
  * **Pipeline Configuration:**
      * **Atom Selection:** Supports C3' (default), P, C4', C5', and O3'.
      * **Parameters:** tunable **Max Distance** ($d_{max}$, default 20Å) and **Bin Width** (default 1.0Å).
  * **Interactive Visualisation:**
      * **Engine:** Built on **Plotly** for dynamic inspection and because it looks cool :).
      * **Modes:**
          * **Combined Overlay:** All 10 base pairs plotted on a single axis for comparison.
          * **Grid View:** A 2x5 subplot matrix isolating specific interactions (ex. AA vs. GC).
  * **Real-time Scoring:**
      * Upload a target PDB to receive an instant pseudo-energy score.
      * Returns the total Gibbs free energy estimate and the count of interactions used.

-----

## Shared Architecture

`rna_utils.py`

This module centralizes all the shared logic and functions between the scripts, ensuring that they have the same behaviour when needed. It includes:

* **Robust Parsing**: Handles PDB parsing complexities, including extracting specific atoms (C3'), handling alternate locations, and stopping after the first model (essential for NMR structures).
* **Consistent Geometry**: Defines the logic for valid base pairs (sequence separation $|i-j| \ge 4$) and distance binning in one place.
**Math & I/O**: Contains the linear interpolation function and the generic file loader (load_pair_data) used by both the scorer and the plotter.

`config.py`

Acts as the central source for file paths:

* Defines absolute paths based on the script's location.

* Ensures that train_potential.py, score_structure.py, and run_pipeline.py all automatically find the data/ directory, regardless of where the command is executed from

-----

## Usage: Command Line Pipeline

For batch processing or server-side automation, the original Python scripts remain fully functional.

### **Option A: Run the Full Pipeline**

The `run_pipeline.py` script orchestrates the entire flow: Training $\rightarrow$ Plotting $\rightarrow$ Scoring.

```bash
python run_pipeline.py
```

*Note: This utilizes paths defined in `config.py` (if present) or internal defaults.*

You can also do a custom run with other variables (here using Phosphorus atoms, 25Å cutoff, and saving to a new folder).

```bash
python run_pipeline.py --atom P --max_dist 25.0 --out_dir results/phosphorus_experiment
```

### **Option B: Run Individual Modules**

#### **1. Training**

Calculates distance distributions for 10 base pairs (AA, AU, etc.) and a reference state (XX).

  * **Algorithm:** Iterates through all atom pairs $(i, j)$ where sequence separation $|i-j| \ge 4$.
  * **Smoothing:** Applies a pseudocount ($\epsilon = 1e^{-12}$) to prevent singularities during log calculations.
  * **Clipping:** Scores are clamped to the range $[-10.0, 10.0]$ to handle rare events.

<!-- end list -->

```bash
python train_potential.py --pdb_dir data/pdb_training --out_dir data/potentials_multi --verbose
```

#### **2. Visualisation**

Generates static images of the potentials using Matplotlib.

  * **Outputs:** Generates both a combined overlay plot and a 2x5 grid layout for detailed inspection. Also ouputs an histogram with raw counts of observed distances (to verify data quality).

<!-- end list -->

```bash
python plot_potentials.py --in_dir data/potentials_multi_
```

#### **3. Scoring**

Evaluates a target structure by summing the pseudo-energy of all valid pairwise interactions.

  * **Interpolation:** Uses linear interpolation between bin centers to provide continuous scoring for distances falling between defined bin intervals.

<!-- end list -->

```bash
python score_structure.py --pdb_dir data/pdb_training --pot_dir data/potentials_multi
```

-----

## Methodology

### The Inverse Boltzmann Principle

The pipeline assumes that frequently observed structural features correspond to low-energy states. We calculate a pseudo-energy ($E$) for a base pair type $ab$ at distance $r$ using the inverse Boltzmann relation:

$$
E_{ab}(r) = -kT \ln \left( \frac{P_{obs}(r | ab)}{P_{ref}(r)} \right)
$$

Where:

  * **$P_{obs}(r | ab)$**: The observed probability of finding pair type $ab$ at distance $r$ in the training set.
  * **$P_{ref}(r)$**: The reference probability derived from a "pooled" state (all residues treated indistinguishably).

### Technical Details

  * **Sequence Separation:** Only pairs $(i, j)$ with $|i - j| \ge 4$ are counted to capture tertiary packing rather than local secondary structure constraints.
  * **Reference State:** The reference state is constructed by aggregating all observed pairs regardless of identity, normalizing by the total count of all pairs.
  * **Interpolation Logic:**
      * If $d \le 0$: Use first bin score.
      * If $d \ge d_{max}$: Use last bin score.
      * Else: Linearly interpolate between the centers of the two enclosing bins.

-----

## Results Interpretation

* **High Positive Score (+10):** Unfavorable distance (e.g., steric clash < 4Å).
* **Negative Score (Wells):** Favorable distance (e.g., at ~5-6Å).
* **Jagged Plots:** If plots look spiky or flat, the training dataset is too small. Add more PDB files to smooth out the curves.

-----

## Requirements

  * **Python 3.11+**
  * **Streamlit** (Web Interface)
  * **Plotly** (Interactive Graphing)
  * **Matplotlib** (Static Graphing)
  * **NumPy** (Math operations)

To install all dependencies:

```bash
pip install streamlit plotly matplotlib pandas numpy streamlit-option-menu streamlit-lottie
```