import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import hashlib

CODE = Path(__file__).parents[1] / 'code'
sys.path.insert(0, str(CODE))

class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('bootstrap_inputs'), 'bootstrap input implementation missing')
        import bootstrap_inputs
        self.mod = bootstrap_inputs

    def test_safe_copy_hash_and_identical_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'out'; source = Path(temp) / 'source.txt'
            source.write_text('original', encoding='utf8')
            dest = root / 'code' / 'copy.txt'
            first = self.mod.copy_verified(source, dest, root)
            second = self.mod.copy_verified(source, dest, root)
            self.assertEqual(first['sha256'], second['sha256'])
            self.assertEqual(dest.read_text(), 'original')

    def test_copy_rejects_outside_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'out'; source = Path(temp) / 'source.txt'
            source.write_text('original')
            with self.assertRaises(ValueError):
                self.mod.copy_verified(source, Path(temp) / 'other.txt', root)

    def test_copy_never_overwrites_different_existing_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'out'; root.mkdir(); source = Path(temp) / 'source.txt'
            source.write_text('new'); dest = root / 'existing.txt'; dest.write_text('keep')
            with self.assertRaises(FileExistsError):
                self.mod.copy_verified(source, dest, root)
            self.assertEqual(dest.read_text(), 'keep')

    def test_event_equality_ignores_order_but_not_time(self):
        self.assertTrue(self.mod.events_equal([.2,.1],[2,1],[.1,.2],[1,2]))
        self.assertFalse(self.mod.events_equal([.2,.1],[2,1],[.1,.200001],[1,2]))

    def test_event_equality_rejects_invalid_arrays(self):
        for t,c in [([np.nan],[1]),([.1],[65]),([.1],[1.5]),([.1,.2],[1])]:
            with self.assertRaises(ValueError):
                self.mod.events_equal(t,c,[.1],[1])

    def test_code_copy_selection_includes_powershell_and_excludes_pycache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'run.ps1').write_text('original')
            (root / 'helper.txt').write_text('original')
            (root / '__pycache__').mkdir()
            (root / '__pycache__' / 'ignored.pyc').write_bytes(b'cache')
            selected = [p.relative_to(root).as_posix() for p in self.mod.iter_original_code_files(root)]
            self.assertEqual(selected, ['helper.txt', 'run.ps1'])

    def test_raw_source_must_be_file_below_approved_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'raw'; root.mkdir()
            inside = root / 'input.spk'; inside.write_bytes(b'spk')
            outside = Path(temp) / 'outside.spk'; outside.write_bytes(b'spk')
            self.assertEqual(self.mod.validate_raw_source(inside, root), inside.resolve())
            with self.assertRaises(ValueError):
                self.mod.validate_raw_source(outside, root)
            with self.assertRaises(FileNotFoundError):
                self.mod.validate_raw_source(root / 'missing.spk', root)

    def test_reuse_allowlist_rejects_unknown_record_variant_and_duplicate(self):
        retained = {'WT_DIV14_P1_A1'}
        valid = [
            ('binary_25ms', 'WT_DIV14_P1_A1_25ms.mat'),
            ('binary_25ms', 'WT_DIV14_P1_A1_25ms.npz'),
            ('direct_checkpoints', 'WT_DIV14_P1_A1.json'),
        ]
        self.assertEqual(len(self.mod.validate_reuse_artifacts(valid, retained, require_complete=False)), 3)
        for bad in [
            [('direct_checkpoints', 'STALE_P1_X.json')],
            [('direct_checkpoints', 'WT_DIV14_P1_A1_extra.json')],
            [('unknown_folder', 'WT_DIV14_P1_A1.json')],
            valid + [('direct_checkpoints', 'WT_DIV14_P1_A1.json')],
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.mod.validate_reuse_artifacts(bad, retained, require_complete=False)

    def test_duration_bins_are_derived_and_expected_distribution_enforced(self):
        rows = []
        for plate, duration in [(1, 300.0), (3, 299.75)]:
            for div in range(14, 64, 7):
                for well in range(6):
                    rows.append({'recording': f'WT_DIV{div}_P{plate}_W{well}', 'DIV': div,
                                 'plate': plate, 'duration_s': duration if plate == 3 and div in {28,63} else 300.0})
        evidence = self.mod.validate_duration_bins(pd.DataFrame(rows))
        self.assertEqual(evidence['bin_count_distribution'], {'11990': 12, '12000': 84})
        rows[-1]['duration_s'] = 299.0
        with self.assertRaises(ValueError):
            self.mod.validate_duration_bins(pd.DataFrame(rows))

    def test_candidate_selection_ignores_legitimate_p3_but_rejects_stale_p1p2(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder, suffixes in self.mod.REUSE_VARIANTS.items():
                target = root / folder; target.mkdir()
                for suffix in suffixes:
                    (target / ('WT_DIV14_P1_A1' + suffix)).write_bytes(b'keep')
                    (target / ('WT_DIV14_P3_A1' + suffix)).write_bytes(b'ignore')
            selected = self.mod.select_reuse_candidates(root, {'WT_DIV14_P1_A1'}, require_complete=True)
            self.assertEqual(len(selected), sum(map(len, self.mod.REUSE_VARIANTS.values())))
            stale = root / 'direct_checkpoints' / 'WT_DIV14_P1_Z9.json'
            stale.write_bytes(b'stale')
            with self.assertRaises(ValueError):
                self.mod.select_reuse_candidates(root, {'WT_DIV14_P1_A1'}, require_complete=True)

    def test_validator_rejects_wrong_root_same_basename(self):
        import validate_bootstrap
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); source_root = base/'source'; destination_root = base/'destination'
            wrong = base/'wrong'; source_root.mkdir(); destination_root.mkdir(); wrong.mkdir()
            source = source_root/'record.json'; destination = wrong/'record.json'
            source.write_bytes(b'x'); destination.write_bytes(b'x')
            digest = hashlib.sha256(b'x').hexdigest()
            manifest = base/'manifest.csv'
            pd.DataFrame([{'source':source,'destination':destination,'sha256':digest,'bytes':1}]).to_csv(manifest,index=False)
            with self.assertRaises(ValueError):
                validate_bootstrap.verify_manifest(manifest, source_root, destination_root, True)

    def test_validator_requires_exact_retained_grid_and_metadata(self):
        import validate_bootstrap
        rows = []
        for div in range(14,64,7):
            for plate in (1,2,3):
                for well in ('A1','A2','A3','B1','B2','B3'):
                    rid=f'WT_DIV{div}_P{plate}_{well}'
                    if rid != self.mod.EXCLUDED:
                        rows.append({'recording':rid,'DIV':div,'plate':plate,'well':well})
        valid = pd.DataFrame(rows)
        self.assertEqual(len(validate_bootstrap.validate_retained_grid(valid)), 95)
        forged = valid.copy()
        forged.loc[forged.index[0], 'recording'] = 'WT_DIV99_P1_A1'
        with self.assertRaises(ValueError):
            validate_bootstrap.validate_retained_grid(forged)
        inconsistent = valid.copy()
        inconsistent.loc[inconsistent.index[0], 'DIV'] = 99
        with self.assertRaises(ValueError):
            validate_bootstrap.validate_retained_grid(inconsistent)

if __name__ == '__main__':
    unittest.main()
