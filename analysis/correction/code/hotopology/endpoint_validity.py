"""Apply the author-approved undefined endpoint-score policy at export."""
import numpy as np
import pandas as pd

def mark_undefined_endpoint_z(table: pd.DataFrame) -> pd.DataFrame:
    result=table.copy()
    missing=(result['metric'].eq('h1_infinite_count') & result['null_sd'].eq(0) & result['observed'].ne(result['null_mean']))
    result.loc[missing,'z']=np.nan
    result['z_available']=np.isfinite(result['z'])
    result['z_unavailable_reason']=np.where(missing,'zero_null_variance_observed_differs','')
    return result
