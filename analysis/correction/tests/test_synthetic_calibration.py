import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE = Path(__file__).parents[1] / 'code'
sys.path.insert(0, str(CODE))


def module():
    spec = importlib.util.find_spec('calibration.run_synthetic_calibration')
    assert spec is not None, 'synthetic calibration module must exist'
    from calibration import run_synthetic_calibration
    return run_synthetic_calibration


def test_grid_and_seeds_are_complete_independent_and_reproducible():
    m = module()
    cells = m.design_grid(['p1','p2','p3','p4','p5','p6'])
    assert len(cells) == 240
    assert len({(c['recording'],c['family'],c['dataset_index']) for c in cells}) == 240
    seeds = [v for c in cells for v in c['seeds'].values()]
    assert len(seeds) == len(set(seeds))
    assert cells == m.design_grid(['p1','p2','p3','p4','p5','p6'])
    assert all(len([c for c in cells if c['recording']==p and c['family']==f])==20
               for p in ['p1','p2','p3','p4','p5','p6'] for f in m.FAMILIES)


def test_summary_keeps_failures_missing_and_wilson_denominators():
    m = module()
    rows = pd.DataFrame([dict(recording='p', family='kinetic', arm='oracle',
        completed=True,p_upper=.01,observed_mass=1,z=2,node_rmse=.1,pair_rmse=.2,lag1_rmse=.3),
        dict(recording='p',family='kinetic',arm='oracle',completed=False,p_upper=np.nan,
             observed_mass=np.nan,z=np.nan,node_rmse=np.nan,pair_rmse=np.nan,lag1_rmse=np.nan)])
    out=m.summarize(rows, requested_per_prototype=2)
    one=out.loc[out.recording.eq('p')].iloc[0]
    assert one.completed==1 and one.failed==1 and one.n_reject==1
    assert one.rejection_fraction==1 and 0<one.wilson_lower<1
    assert np.allclose(m.wilson_interval(0,100), (0, .0369934982), atol=1e-8)
    assert all(np.isnan(m.wilson_interval(0,0)))


def test_scoring_preserves_counts_k_and_lag_time_boundaries():
    m=module()
    rng=np.random.default_rng(3)
    x=rng.integers(0,2,(4,80),dtype=np.uint8)
    sims=np.stack([x,x[:,::-1],np.roll(x,1,axis=1)])
    result=m.score_simulations(x,sims)
    assert result['null_counts'].shape==(3,4)
    assert np.array_equal(result['observed_counts'], result['null_counts'][0])
    assert result['exact_k_mismatch_count']==np.count_nonzero(sims.sum(1)!=x.sum(0))
    expected=sum(s[:,1:].astype(float)@s[:,:-1].T/79 for s in sims)/3
    target=x[:,1:].astype(float)@x[:,:-1].T/79
    assert np.isclose(result['lag1_rmse'],np.sqrt(np.mean((expected-target)**2)))
    assert 1/4<=result['p_upper']<=1


def test_synthetic_wrapper_seeds_native_sampler_and_uses_original_fit():
    path=CODE/'calibration/fit_synthetic_maxent.m'
    assert path.exists(), 'native seeded synthetic wrapper must exist'
    source=path.read_text()
    assert 'fit_maxent_recording(' in source
    assert "'randseed', sampleSeed" in source
    assert "rng(double(fitSeed), 'twister')" in source
    assert '20, size(x, 2)' in source


def test_dataset_failure_retains_both_requested_arms_as_missing():
    m=module()
    cell=m.design_grid(['WT_DIV14_P1_A2'])[0]
    assert hasattr(m,'dataset_failure_rows'), 'generation/load failures need explicit missing arm rows'
    rows=m.dataset_failure_rows(cell,'source unavailable')
    assert len(rows)==2 and {r['arm'] for r in rows}=={'oracle','refit'}
    assert all(not r['completed'] and r['reject'] is None and np.isnan(r['p_upper']) for r in rows)
    assert all(r['error']=='source unavailable' for r in rows)


def test_incompatible_checkpoint_is_preserved_before_replacement(tmp_path):
    import json
    m=module()
    assert hasattr(m,'archive_incompatible'), 'incompatible checkpoints must remain auditable'
    (tmp_path/'provenance.json').write_text(json.dumps({'input_hash':'old'}))
    (tmp_path/'oracle.npz').write_bytes(b'old result')
    m.archive_incompatible(tmp_path,'new')
    assert (tmp_path/'history/old/oracle.npz').read_bytes()==b'old result'


def test_nonconverged_fit_is_invalid_and_environment_retry_is_explicit():
    import pytest
    m=module()
    assert hasattr(m,'validate_fit'), 'nonconverged fit must not produce a valid score'
    with pytest.raises(RuntimeError, match='nonconverged'):
        m.validate_fit({'convergence':False})
    m.validate_fit({'convergence':True})
    assert hasattr(m,'is_environment_failure')
    assert m.is_environment_failure('MATLAB license checkout failed')
    assert not m.is_environment_failure('nonconverged original refit')


def test_localized_5001_retry_adapter_preserves_original_failure(tmp_path):
    import json
    m=module()
    path=CODE.parents[0]/'calibration/synthetic/reclassify_matlab_5001.py'
    assert path.exists(), 'localized MATLAB 5001 requires an independently audited retry adapter'
    spec=importlib.util.spec_from_file_location('retry5001',path)
    adapter=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    original='MATLAB exit 1: \\x7f MathWorks https://www.mathworks.com/support/lme/5001'
    rows=[dict(arm='oracle',completed=True,error='',input_hash='frozen'),
          dict(arm='refit',completed=False,error=original,input_hash='frozen',failure_kind='model_or_scoring')]
    (tmp_path/'provenance.json').write_text(json.dumps({'input_hash':'frozen'}))
    (tmp_path/'rows.json').write_text(json.dumps(rows))
    changed=adapter.reclassify_cell(tmp_path)
    assert changed==1
    updated=json.loads((tmp_path/'rows.json').read_text())
    assert updated[1]['original_error']==original
    assert updated[1]['failure_kind']=='environment'
    assert m.is_environment_failure(updated[1]['error'])
    assert json.loads((tmp_path/'history/frozen/rows.json').read_text())==rows
    assert adapter.reclassify_cell(tmp_path)==0


def audit_module():
    path=CODE.parents[0]/'calibration/synthetic/audit_completed_outputs.py'
    assert 'def validate_counts(' in path.read_text(), 'audit must fail closed on invariant violations'
    spec=importlib.util.spec_from_file_location('synthetic_audit',path)
    audit=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    return audit


def test_audit_fails_on_generation_k_or_initial_state_violation():
    audit=audit_module()
    good=dict(recorded_scores=480,completed_scores=480,failed_scores=0,
        independently_rescored=480,generated_datasets_audited=240,
        allocated_unique_stage_seeds=1200,generation_k_mismatches=0,
        generation_initial_state_mismatches=0,reference_k_mismatches_reported=0)
    assert audit.validate_counts(good)==[]
    for key in ('generation_k_mismatches','generation_initial_state_mismatches',
                'reference_k_mismatches_reported'):
        assert audit.validate_counts(dict(good,**{key:1})), key
    assert audit.validate_counts(dict(good,independently_rescored=479))


def test_audit_rejects_stale_aggregate_count_and_wilson_interval():
    audit=audit_module(); m=module()
    rows=pd.DataFrame([dict(recording='p',family='kinetic',arm='oracle',completed=True,
        p_upper=.01,observed_mass=.2,z=1,node_rmse=.1,pair_rmse=.1,lag1_rmse=.1)])
    aggregate=m.summarize(rows)
    assert audit.validate_aggregate(rows,aggregate)==[]
    stale=aggregate.copy(); stale.loc[0,'n_reject']=0
    assert any('n_reject' in error for error in audit.validate_aggregate(rows,stale))
    stale=aggregate.copy(); stale.loc[0,'wilson_lower']+=.1
    assert any('wilson_lower' in error for error in audit.validate_aggregate(rows,stale))


def test_audit_requires_native_fit_and_nonmissing_true_convergence(tmp_path):
    import pytest
    audit=audit_module()
    assert hasattr(audit,'check_refit_artifacts'), 'valid refit requires strict convergence and fit artifacts'
    row=pd.Series(dict(arm='refit',family='instantaneous',completed=True,convergence=True))
    with pytest.raises(AssertionError,match='matlab_fit'):
        audit.check_refit_artifacts(row,tmp_path)
    (tmp_path/'matlab_fit.mat').write_bytes(b'presence fixture; contents validated by full audit')
    audit.check_refit_artifacts(row,tmp_path)
    for invalid in (np.nan,False,'True',1):
        row['convergence']=invalid
        with pytest.raises(AssertionError,match='convergence'):
            audit.check_refit_artifacts(row,tmp_path)
