import numpy as np
import pandas as pd
from hotopology.endpoint_validity import mark_undefined_endpoint_z

def test_only_undefined_endpoint_z_is_missing_and_observation_preserved():
    x=pd.DataFrame({'metric':['h1_infinite_count']*3+['h1_total_persistence_capped'], 'observed':[1.,0.,1.,2.], 'null_mean':[0.,0.,.5,1.], 'null_sd':[0.,0.,.2,.5], 'z':[1e12,0.,2.5,2.], 'empirical_p':[1/101,1.,.1,.2]})
    y=mark_undefined_endpoint_z(x)
    assert np.isnan(y.loc[0,'z'])
    assert y.loc[0,'empirical_p']==x.loc[0,'empirical_p']
    assert y.loc[0,'observed']==1 and y.loc[0,'null_sd']==0
    assert y.loc[1:,'z'].tolist()==[0.,2.5,2.]
    assert y.z_available.tolist()==[False,True,True,True]
    assert x.loc[0,'z']==1e12

def test_reapplication_is_idempotent():
    x=pd.DataFrame({'metric':['h1_infinite_count'],'observed':[1.],'null_mean':[0.],'null_sd':[0.],'z':[np.nan]})
    a=mark_undefined_endpoint_z(x)
    pd.testing.assert_frame_equal(a,mark_undefined_endpoint_z(a))
