import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, **kwargs):
    return subprocess.run([sys.executable, *cmd], text=True, capture_output=True, **kwargs)


def main():
    print('Running simple functional tests (no pytest required)')

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / 'pots'
        print('Training into', out_dir)
        r = run(['train_potential.py', '--pdb_dir', 'data/pdb_training', '--out_dir', str(out_dir)])
        print('train stdout:\n', r.stdout)
        print('train stderr:\n', r.stderr)
        if r.returncode != 0:
            print('TRAIN FAILED')
            sys.exit(2)

        # verify files
        expected = [f'potential_{p}.txt' for p in [
            'AA','AU','AC','AG','UU','UC','UG','CC','CG','GG']]
        for e in expected:
            if not (out_dir / e).exists():
                print('Missing expected file:', e)
                sys.exit(3)

        # run scoring
        pdb = 'data/pdb_training/1EHZ.pdb'
        # Instead of calling the script as a subprocess (some environments
        # don't always forward output or may buffer), import the module and
        # run the same logic directly so we can inspect results programmatically.
        import importlib.util
        spec = importlib.util.spec_from_file_location('score_mod', 'score_structure.py')
        score_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(score_mod)

        pots = score_mod.load_potentials(str(out_dir))
        chains = score_mod.parse_c3_atoms(pdb)

        total_score = 0.0
        n_pairs_used = 0
        for chain_id, residues in chains.items():
            n = len(residues)
            for i in range(n):
                idx_i, res_i, coord_i = residues[i]
                for j in range(i + 4, n):
                    idx_j, res_j, coord_j = residues[j]
                    d = score_mod.distance(coord_i, coord_j)
                    if d > score_mod.MAX_DIST:
                        continue
                    pk = score_mod.pair_key(res_i, res_j)
                    if pk not in pots:
                        continue
                    score = score_mod.interpolate_score(d, pots[pk])
                    total_score += score
                    n_pairs_used += 1

        print('scored pairs:', n_pairs_used)
        print('total pseudo-energy:', total_score)
        if n_pairs_used == 0:
            print('No pairs were scored — failing test')
            sys.exit(5)

    print('All simple tests passed')


if __name__ == '__main__':
    main()
