import subprocess
import sys
from pathlib import Path


def run(cmd, **kwargs):
    return subprocess.run([sys.executable, *cmd], text=True, capture_output=True, **kwargs)


def test_training_writes_potentials(tmp_path: Path):
    out_dir = tmp_path / "potentials_out"
    out_dir_str = str(out_dir)

    # Run training on the example dataset
    r = run(["train_potential.py", "--pdb_dir", "data/pdb_training", "--out_dir", out_dir_str])
    assert r.returncode == 0, f"train_potential failed: {r.stderr}\n{r.stdout}"

    # Expect 10 potential files + summary
    expected_pairs = [
        "AA", "AU", "AC", "AG",
        "UU", "UC", "UG",
        "CC", "CG", "GG",
    ]

    for p in expected_pairs:
        f = out_dir / f"potential_{p}.txt"
        assert f.exists(), f"Missing {f}"
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        assert len(lines) == 20, f"{f} must have 20 lines (has {len(lines)})"

    summary = out_dir / "summary.txt"
    assert summary.exists(), "summary.txt missing"
    text = summary.read_text()
    assert "pdb_files_processed" in text


def test_scoring_uses_potentials(tmp_path: Path):
    # First train to tmp dir
    out_dir = tmp_path / "potentials_out2"
    r = run(["train_potential.py", "--pdb_dir", "data/pdb_training", "--out_dir", str(out_dir)])
    assert r.returncode == 0

    # Now run scoring on the sample PDB using generated potentials
    pdb = "data/pdb_training/1EHZ.pdb"
    r2 = run(["score_structure.py", "--pdb", pdb, "--pot_dir", str(out_dir)])
    assert r2.returncode == 0
    out = r2.stdout + r2.stderr
    assert "Estimated pseudo-energy" in out or "Estimated pseudo-energy" in r2.stdout
    assert "Number of C3'-C3' pairs used" in out