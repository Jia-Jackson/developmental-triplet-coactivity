import numpy as np
from hotopology.toolbox_cache import can_reuse_energy_sign_cache

def test_nontrivial_missing_k_invalidates_old_cache():
    pool=np.zeros((3, 5), dtype=np.uint8)
    assert not can_reuse_energy_sign_cache({}, pool, np.array([0,1]))
    assert can_reuse_energy_sign_cache({'toolbox_logpotential_version':2}, pool, np.array([0,1]))

def test_no_fallback_or_unique_endpoint_states_are_unchanged():
    pool=np.zeros((3,5), dtype=np.uint8)
    assert can_reuse_energy_sign_cache({}, pool, np.array([0]))
    assert can_reuse_energy_sign_cache({}, pool, np.array([0,3]))
