"""Frozen 6 x 2 x 20 synthetic generator/refit/oracle experiment (100 nulls)."""
from __future__ import annotations

from release_paths import repository_path

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from calibration.exact_pairwise import ExactPairwise, triplet_residual_score
from calibration.run_observed_benchmark import empirical_triplet_support_zeta, theta_from_toolbox_factors
from hotopology.kinetic_null import fit_kinetic_pairwise, simulate_kinetic_conditioned_k
from hotopology.null_models import decode_ising_factors, resample_pool_conditioned_on_k

FAMILIES = ('instantaneous', 'kinetic')
ARMS = ('oracle', 'refit')
N_NULL = 100
VERSION = 1
STAGES = ('generation', 'oracle', 'refit', 'fit', 'source_pool')
SCALARS = ('observed_mass', 'null_mass_mean', 'null_mass_sd', 'z', 'p_upper',
           'status', 'node_rmse', 'pair_rmse', 'lag1_rmse', 'exact_k_mismatch_count')


def design_grid(prototypes):
    cells = [dict(recording=p, family=f, dataset_index=i,
                  seeds={s: zlib.crc32(f'synthetic_v1|{p}|{f}|{i}|{s}'.encode())
                         for s in STAGES})
             for p in prototypes for f in FAMILIES for i in range(20)]
    seeds = [v for c in cells for v in c['seeds'].values()]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError('Actual CRC32 seed collision')
    return cells


def wilson_interval(rejections, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.95996398454
    p = rejections / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0., center-half), min(1., center+half)


def summarize(rows, requested_per_prototype=20):
    result = []
    groups = [(str(p), f, a, g, requested_per_prototype)
              for (p,f,a),g in rows.groupby(['recording','family','arm'])]
    groups += [('POOLED',f,a,g, requested_per_prototype*g.recording.nunique())
               for (f,a),g in rows.groupby(['family','arm'])]
    for p,f,a,g,requested in groups:
        ok = g.loc[g.completed.astype(bool) & np.isfinite(g.p_upper)]
        n = len(ok)
        r = int((ok.p_upper <= .05).sum())
        lo,hi = wilson_interval(r,n)
        record = dict(recording=p,family=f,arm=a,requested=requested,completed=n,
                      failed=requested-n,n_reject=r,rejection_fraction=r/n if n else np.nan,
                      wilson_lower=lo,wilson_upper=hi,inflation_flag=bool(lo>.05))
        for key in ('observed_mass','z','node_rmse','pair_rmse','lag1_rmse'):
            v = ok[key].to_numpy(float)
            v = v[~np.isnan(v)]
            for q,label in ((.25,'q25'),(.5,'median'),(.75,'q75')):
                # Preserve infinite z as diagnostic, without interpolation of infinities.
                record[key+'_'+label] = float(np.quantile(v,q,method='nearest')) if len(v) else np.nan
        result.append(record)
    return pd.DataFrame(result)


def score_simulations(x, simulated):
    counts, support = empirical_triplet_support_zeta(x)
    pairs = np.triu_indices(x.shape[0],1)
    null_counts=[]; null_support=[]
    moments=[]; lag=[]
    for sim in simulated:
        c,s=empirical_triplet_support_zeta(sim)
        null_counts.append(c); null_support.append(s)
        d=sim.astype(float)
        moments.append(np.concatenate((d.mean(1),(d@d.T/d.shape[1])[pairs])))
        lag.append(d[:,1:]@d[:,:-1].T/(d.shape[1]-1))
    null_support=np.array(null_support)
    result=triplet_residual_score(support,null_support)
    d=x.astype(float)
    obs_moments=np.concatenate((d.mean(1),(d@d.T/d.shape[1])[pairs]))
    error=np.mean(moments,axis=0)-obs_moments
    return dict(observed_counts=counts,observed_support=support,null_counts=np.array(null_counts),
        null_supports=null_support,null_scores=result['null_scores'],observed_mass=result['score'],
        null_mass_mean=result['null_mean'],null_mass_sd=result['null_sd'],z=result['z'],
        p_upper=result['p_upper'],status=result['status'],
        node_rmse=float(np.sqrt(np.mean(error[:x.shape[0]]**2))),
        pair_rmse=float(np.sqrt(np.mean(error[x.shape[0]:]**2))),
        lag1_rmse=float(np.sqrt(np.mean((np.mean(lag,axis=0)-d[:,1:]@d[:,:-1].T/(d.shape[1]-1))**2))),
        exact_k_mismatch_count=int(np.count_nonzero(simulated.sum(1)!=x.sum(0))))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump_json(path,payload):
    Path(path).write_text(json.dumps(payload,indent=2,default=str),encoding='utf8')


def dataset_failure_rows(cell, error):
    return [dict(recording=cell['recording'],family=cell['family'],dataset_index=cell['dataset_index'],
                 arm=arm,completed=False,reject=None,error=error,status='failed',
                 **{key:np.nan for key in SCALARS if key!='status'}) for arm in ARMS]


def validate_fit(metadata):
    if metadata.get('convergence') is False:
        raise RuntimeError('nonconverged original refit: score unavailable')


def is_environment_failure(error):
    return any(term in str(error).lower() for term in
               ('license','licensing','failed to start','connection','permissionerror'))


def archive_incompatible(folder, input_hash, force=False):
    provenance=folder/'provenance.json'
    if not provenance.exists(): return
    old_hash=json.loads(provenance.read_text())['input_hash']
    if old_hash==input_hash and not force: return
    dest=folder/'history'/old_hash
    if dest.exists(): dest=folder/'history'/f'{old_hash}_{time.time_ns()}'
    dest.mkdir(parents=True)
    for path in folder.iterdir():
        if path.is_file(): shutil.copy2(path,dest/path.name)


def load_prototype(root,recording):
    mp=root/'data_intermediate/maxent_models_top16'/f'{recording}_maxent.mat'
    dp=root/'data_intermediate/binary_25ms'/f'{recording}_25ms.npz'
    kp=root/'data_intermediate/kinetic_models_top16'/f'{recording}_kinetic_fit.npz'
    mat=loadmat(mp,squeeze_me=True,struct_as_record=False)
    nodes=np.atleast_1d(mat['active_nodes']).astype(int)-1
    with np.load(dp) as z: x=z['x'][nodes].astype(np.uint8)
    with np.load(kp) as z:
        if not np.array_equal(z['active_nodes_zero_based'],nodes):
            raise RuntimeError('Prototype kinetic and instantaneous nodes differ')
        intercepts=z['intercepts']; weights=z['weights']
    if x.shape[0]!=16 or x.shape[1] not in (11990,12000):
        raise RuntimeError('Frozen prototype shape gate failed')
    return dict(x=x,nodes=nodes,theta=theta_from_toolbox_factors(mat['fit'].pairwise_model.factors,16),
                intercepts=intercepts,weights=weights,
                source_sha={str(p):sha(p) for p in (mp,dp,kp)})


def fit_instantaneous(root,folder,x,seeds,matlab,toolbox,input_hash):
    inp=folder/'matlab_input.mat'; out=folder/'matlab_fit.mat'
    marker=folder/'matlab_fit_provenance.json'
    reuse=out.exists() and marker.exists() and json.loads(marker.read_text())['input_hash']==input_hash
    if not reuse:
        savemat(inp,dict(x=x))
        def mq(p): return str(p).replace('\\','/').replace("'","''")
        command=(f"restoredefaultpath; addpath('{mq(root/'code')}'); "
            f"addpath('{mq(root/'code/calibration')}'); "
            f"fit_synthetic_maxent('{mq(inp)}','{mq(out)}','{mq(toolbox)}',"
            f"{seeds['fit']},{seeds['source_pool']});")
        prefs=root/'work'/'synthetic_matlab_prefs'/folder.name
        prefs.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy(); env['MATLAB_PREFDIR']=str(prefs)
        for attempt in range(2):
            log=folder/f'matlab_attempt_{attempt}.log'
            started=time.time()
            with log.open('w',encoding='utf8') as f:
                p=subprocess.run([str(matlab),'-nosplash','-nodesktop','-batch',command],
                    stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            text=log.read_text(encoding='utf8',errors='replace')
            dump_json(folder/f'matlab_attempt_{attempt}.json',dict(exit_code=p.returncode,
                elapsed_seconds=time.time()-started,input_hash=input_hash,fit_seed=seeds['fit'],
                sample_seed=seeds['source_pool']))
            if p.returncode==0 and out.exists(): break
            environment_failure=is_environment_failure(text)
            if not environment_failure or attempt==1: raise RuntimeError(f'MATLAB exit {p.returncode}: {text[-1500:]}')
        dump_json(marker,dict(input_hash=input_hash,source_pool_seed=seeds['source_pool']))
    mat=loadmat(out,squeeze_me=True,struct_as_record=False)
    fit=mat['fit']
    if not (bool(mat['native_seed_repeat_equal']) and bool(mat['native_seed_change_different'])):
        raise RuntimeError('Native RNG validation failed')
    h,j=decode_ising_factors(fit.pairwise_model.factors,x.shape[0])
    pool=np.asarray(fit.pairwise_samples,dtype=np.uint8).reshape(x.shape[0],-1)
    wanted=np.unique(x.sum(0)); existing=np.unique(pool.sum(0))
    missing=np.setdiff1d(wanted,existing).astype(int).tolist()
    metadata=dict(convergence=bool(fit.pairwise_converged),iterations=None,
        iteration_availability='Original MATLAB API does not return iteration count',
        fit_warning=str(mat.get('fit_warning','')),missing_k_buckets=json.dumps(missing),
        missing_k_bin_count=int(np.isin(x.sum(0),missing).sum()),
        source_pool_bins=pool.shape[1],native_seed_repeat_equal=True,native_seed_change_different=True,
        source_pool_node_rmse=float(fit.pairwise_firing_rmse),source_pool_pair_rmse=float(fit.pairwise_pair_rmse),
        source_pool_k_rmse=float(fit.pairwise_k_rmse))
    return pool,h,j,metadata


def run_cell(root,cell,matlab,toolbox,code_sha,retry_environment=False):
    root=Path(root); start=time.time()
    rec=cell['recording']; fam=cell['family']; index=cell['dataset_index']; seeds=cell['seeds']
    folder=root/'calibration/synthetic/cells'/f'{rec}_{fam}_{index:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    prototype=load_prototype(root,rec)
    provenance=dict(version=VERSION,cell=cell,source_sha=prototype['source_sha'],code_sha=code_sha,n_null=N_NULL)
    ih=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    provenance['input_hash']=ih
    completed_path=folder/'rows.json'
    old=[]
    if completed_path.exists():
        old=json.loads(completed_path.read_text())
        if all(r['input_hash']==ih for r in old):
            if not retry_environment or not any(is_environment_failure(r['error']) for r in old): return old
        else: old=[]
    archive_incompatible(folder,ih,force=bool(old))
    dump_json(folder/'provenance.json',provenance)
    obs=prototype['x']; k=obs.sum(0).astype(np.int16)
    if fam=='instantaneous':
        x=ExactPairwise(16).simulate(prototype['theta'],k,1,seeds['generation'])[0]
    else:
        x=simulate_kinetic_conditioned_k(prototype['intercepts'],prototype['weights'],k,
            initial_state=obs[:,0],n_realizations=1,seed=seeds['generation'])[0]
        if not np.array_equal(x[:,0],obs[:,0]): raise RuntimeError('Kinetic initial state mismatch')
    if not np.array_equal(x.sum(0),k): raise RuntimeError('Synthetic generator exact-K mismatch')
    np.savez_compressed(folder/'synthetic.npz',x=x,k=k,active_nodes_zero_based=prototype['nodes'],
        generating_theta=prototype['theta'],generating_intercepts=prototype['intercepts'],
        generating_weights=prototype['weights'],generation_seed=seeds['generation'],input_hash=ih)
    rows=[]
    for arm in ARMS:
        arm_start=time.time()
        base=dict(recording=rec,culture=rec.split('_')[2],DIV=int(rec.split('_')[1][3:]),
            family=fam,dataset_index=index,arm=arm,n_nodes=16,n_bins=x.shape[1],n_null=N_NULL,
            seed=seeds[arm],generation_seed=seeds['generation'],input_hash=ih,
            fit_seed=seeds['fit'],source_pool_seed=seeds['source_pool'],
            convergence=None,iterations=None,fit_warning='',error='',completed=False,
            reject=None,missing_k_buckets='[]',missing_k_bin_count=0)
        checkpoint=folder/f'{arm}.npz'
        previous=next((r for r in old if r['arm']==arm),None)
        if previous and not is_environment_failure(previous['error']):
            rows.append(previous); continue
        if checkpoint.exists():
            with np.load(checkpoint) as z:
                if str(z['input_hash'].item())==ih:
                    rows.append(json.loads(str(z['summary_json'].item()))); continue
        try:
            if fam=='instantaneous':
                if arm=='oracle':
                    simulated=ExactPairwise(16).simulate(prototype['theta'],k,N_NULL,seeds[arm])
                else:
                    pool,h,j,meta=fit_instantaneous(root,folder,x,seeds,Path(matlab),Path(toolbox),ih)
                    base.update(meta)
                    validate_fit(base)
                    simulated=resample_pool_conditioned_on_k(pool,k,n_surrogates=N_NULL,
                        seed=seeds[arm],fallback_h=h,fallback_j=j,fallback_sweeps=20)
            else:
                intercepts=prototype['intercepts']; weights=prototype['weights']
                if arm=='refit':
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        fit=fit_kinetic_pairwise(x,c_grid=(.01,.1,1),n_splits=3,max_iter=500)
                    warning_text=' | '.join(str(w.message) for w in caught)
                    base.update(convergence=not any(w.category.__name__=='ConvergenceWarning' for w in caught),
                        convergence_source='absence of sklearn ConvergenceWarning; iteration counts not exposed',
                        fit_warning=warning_text,heldout_log_loss=fit.heldout_log_loss,
                        rate_only_log_loss=fit.rate_only_log_loss)
                    intercepts=fit.intercepts; weights=fit.weights
                    np.savez_compressed(folder/'kinetic_fit.npz',**vars(fit),input_hash=ih)
                    validate_fit(base)
                simulated=simulate_kinetic_conditioned_k(intercepts,weights,k,
                    initial_state=x[:,0],n_realizations=N_NULL,seed=seeds[arm])
                if not np.all(simulated[:,:,0]==x[None,:,0]): raise RuntimeError('Reference initial state mismatch')
            result=score_simulations(x,simulated)
            if result['null_supports'].shape!=(100,560): raise RuntimeError('Null triplet shape gate failed')
            if result['exact_k_mismatch_count'] or not np.isfinite(result['p_upper']):
                raise RuntimeError('Reference K or finite P gate failed')
            base.update({s:result[s] for s in SCALARS},completed=True,reject=bool(result['p_upper']<=.05),
                        runtime_seconds=time.time()-arm_start)
            np.savez_compressed(checkpoint,input_hash=ih,summary_json=json.dumps(base),
                **{key:result[key] for key in ('observed_counts','observed_support','null_counts',
                                            'null_supports','null_scores')})
            del simulated
        except Exception as exc:
            base.update({s:np.nan for s in SCALARS},status='failed',error=repr(exc),
                        failure_kind='environment' if is_environment_failure(exc) else 'model_or_scoring',
                        runtime_seconds=time.time()-arm_start)
            (folder/f'{arm}_failure.txt').write_text(traceback.format_exc(),encoding='utf8')
        rows.append(base)
        dump_json(folder/f'{arm}_row.json',base)
    dump_json(completed_path,rows)
    print(f'{rec} {fam} {index:02d}: {time.time()-start:.1f}s; completed={sum(r["completed"] for r in rows)}/2',flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result-dir',type=Path,required=True)
    parser.add_argument('--matlab',type=Path,default=Path(r'D:\Application\Matlab\bin\matlab.exe'))
    parser.add_argument('--toolbox',type=Path,default=repository_path('third_party/maxent_toolbox'))
    parser.add_argument('--mode',choices=('pilot','full'),default='full')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--retry-environment',action='store_true',
                        help='Explicit same-seed retry of recorded environment failures; preserve prior attempts')
    args=parser.parse_args(); root=args.result_dir.resolve()
    output=root/'calibration/synthetic'; output.mkdir(parents=True,exist_ok=True)
    candidates=pd.read_csv(root/'calibration/observed_benchmark/candidates.csv')
    selected=candidates.loc[candidates.selected & candidates.DIV.isin([14,63])].sort_values(['culture','DIV'])
    if len(selected)!=6: raise RuntimeError('Expected six frozen selected prototypes')
    cells=design_grid(selected.recording.tolist())
    code=root/'code'
    paths=[Path(__file__),code/'calibration/fit_synthetic_maxent.m',code/'fit_maxent_recording.m',
        code/'calibration/exact_pairwise.py',code/'calibration/run_observed_benchmark.py',
        code/'hotopology/null_models.py',code/'hotopology/kinetic_null.py',
        args.toolbox/'+maxent/generateSamples.m',args.toolbox/'+maxent/trainModel.m',
        args.toolbox/'+maxent/trainModelExhaustive.m',args.toolbox/'mex_code/mexGibbsSampler.cpp']
    sources={str(p):sha(p) for p in paths}
    # Include native compiled binaries in the execution provenance.
    sources.update({str(p):sha(p) for p in args.toolbox.rglob('*.mexw64')})
    dump_json(output/'design.json',dict(version=VERSION,n_null=N_NULL,cells=cells,code_sha=sources,
        prototype_rows=selected.to_dict('records'),python=sys.executable,numpy=np.__version__,
        stages=STAGES,threads_per_process=2,max_workers=min(2,args.workers)))
    if args.mode=='pilot': cells=[c for c in cells if c['recording']==selected.recording.iloc[0] and c['dataset_index']==0]
    started=time.time(); rows=[]
    with ProcessPoolExecutor(max_workers=min(2,args.workers)) as executor:
        futures={executor.submit(run_cell,root,c,args.matlab,args.toolbox,sources,args.retry_environment):c for c in cells}
        for future in as_completed(futures):
            cell=futures[future]
            try: rows.extend(future.result())
            except Exception as exc:
                failed=dataset_failure_rows(cell,repr(exc))
                rows.extend(failed)
                dump_json(output/f"{cell['recording']}_{cell['family']}_{cell['dataset_index']:02d}_failure.json",
                          dict(cell=cell,error=repr(exc),traceback=traceback.format_exc(),code_sha=sources,rows=failed))
            pd.DataFrame(rows).to_csv(output/f'summary_{args.mode}_progress.csv',index=False)
    table=pd.DataFrame(rows).sort_values(['recording','family','dataset_index','arm'])
    table.to_csv(output/f'summary_{args.mode}.csv',index=False)
    aggregate=summarize(table,requested_per_prototype=1 if args.mode=='pilot' else 20)
    aggregate.to_csv(output/f'aggregate_{args.mode}.csv',index=False)
    dump_json(output/f'run_{args.mode}.json',dict(elapsed_seconds=time.time()-started,
        expected_scores=2*len(cells),recorded_scores=len(rows),completed=int(table.completed.sum()),
        failures=int((~table.completed).sum()),unique_stage_seeds=len({s for c in cells for s in c['seeds'].values()})))
    print(aggregate.loc[aggregate.recording.eq('POOLED')].to_string(index=False),flush=True)
    if len(rows)!=2*len(cells): raise RuntimeError('Grid closure failed')
    if args.mode=='pilot' and not table.completed.all(): raise RuntimeError('Pilot failed; inspect recorded errors')


if __name__=='__main__':
    main()
