"""Safe compatibility of pre-correction sample-pool caches."""
import hashlib
from pathlib import Path
import shutil
import numpy as np

def can_reuse_energy_sign_cache(payload, pool, desired_k):
    if payload.get('toolbox_logpotential_version') == 2:
        return True
    samples = np.asarray(pool)
    if samples.ndim != 2:
        raise ValueError('Expected node-by-sample pool')
    missing = set(np.asarray(desired_k, dtype=int).tolist()) - set(samples.sum(axis=0).tolist())
    # K=0 and K=N each have exactly one state, so their output is parameter-independent.
    return not (missing - {0, samples.shape[0]})

def archive_invalid_cache(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    relative = path.relative_to(root)
    target = root / 'audit/pre_energy_sign/checkpoints' / relative
    target.resolve().relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(path.read_bytes()).digest():
            raise ValueError(f'Conflicting archived cache: {target}')
    else:
        shutil.copy2(path, target)
