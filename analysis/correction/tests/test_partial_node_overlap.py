import numpy as np
import pandas as pd
from scipy.io import savemat
from run_v2_node_robustness import _dynamic_direct


def test_unavailable_fixed_overlap_keeps_dynamic_metrics(tmp_path):
    out=tmp_path/'tables/computational_closure_v2'; out.mkdir(parents=True)
    model=tmp_path/'data_intermediate/maxent_models_top16'; model.mkdir(parents=True)
    binary=tmp_path/'data_intermediate/binary_25ms'; binary.mkdir()
    names=['WT_DIV14_P1_A1','WT_DIV14_P3_B3']
    pd.DataFrame([dict(recording=names[0],n_nodes=16,electrode_index_zero_based=i)
                  for i in range(16)]).to_csv(out/'v2_fixed_node_manifest_long.csv',index=False)
    for rec,plate,well in [(names[0],1,'A1'),(names[1],3,'B3')]:
        x=np.zeros((64,60),dtype=np.uint8); x[:16,:20]=1
        np.savez(binary/f'{rec}_25ms.npz',x=x,div=14,plate=plate,well=well)
        savemat(model/f'{rec}_maxent.mat',dict(active_nodes=np.arange(1,17)))
    direct,overlap=_dynamic_direct(tmp_path)
    assert len(direct)==2
    assert overlap.loc[overlap.recording.eq(names[0]),'jaccard'].iloc[0]==1
    assert np.isnan(overlap.loc[overlap.recording.eq(names[1]),'jaccard'].iloc[0])
