"""Observed exact/conditional benchmark for the audited dynamic-N16 models."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from calibration.exact_pairwise import ExactPairwise, triplet_residual_score
from hotopology.null_models import decode_ising_factors, resample_pool_conditioned_on_k


N_NULL = 100
ARMS = ("finite_pool", "exact", "conditional_refit")
PIPELINE_VERSION = 2


def stable_seed(recording: str, arm: str) -> int:
    return int(zlib.crc32((recording + arm).encode("utf-8")))


def theta_from_toolbox_factors(factors: np.ndarray, n_nodes: int) -> np.ndarray:
    """Map decoded log-potential [h, J01, J02, ...] to exact features."""
    h, j = decode_ising_factors(np.asarray(factors), n_nodes)
    return np.concatenate((h, np.fromiter(
        (j[i, k] for i, k in combinations(range(n_nodes), 2)), dtype=float
    )))


def empirical_triplet_support_zeta(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return integer triplet counts and support using a state-histogram zeta transform."""
    data = np.asarray(x)
    if data.ndim != 2 or not np.isin(data, (0, 1)).all():
        raise ValueError("x must be a binary N x T matrix")
    n, n_bins = data.shape
    if not 1 <= n <= 16 or n_bins == 0:
        raise ValueError("x must have 1..16 nonempty node rows")
    ids = np.sum(data.astype(np.uint32) * (1 << np.arange(n, dtype=np.uint32))[:, None], axis=0)
    supersets = np.bincount(ids, minlength=1 << n).astype(np.int64)
    state_ids = np.arange(1 << n, dtype=np.uint32)
    for bit in range(n):
        absent = (state_ids & (1 << bit)) == 0
        supersets[absent] += supersets[state_ids[absent] | (1 << bit)]
    masks = np.fromiter(((1 << i) | (1 << j) | (1 << k)
                         for i, j, k in combinations(range(n), 3)), dtype=np.uint32)
    counts = supersets[masks].copy()
    return counts, counts.astype(float) / n_bins


def select_lower_median_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"culture", "DIV", "recording", "eligible", "mean_population_count", "reason"}
    if not required <= set(candidates.columns):
        raise ValueError(f"missing candidate columns: {sorted(required - set(candidates.columns))}")
    out = candidates.copy()
    out["selected"] = False
    out["candidate_n"] = 0
    out["exclusion_reason"] = np.where(out["eligible"].astype(bool), "", out["reason"].fillna("ineligible"))
    for _, index in out.groupby(["culture", "DIV"], sort=True).groups.items():
        eligible = out.loc[index][out.loc[index, "eligible"].astype(bool)].sort_values(
            ["mean_population_count", "recording"], kind="stable"
        )
        n = len(eligible)
        out.loc[index, "candidate_n"] = n
        if n:
            out.loc[eligible.index[(n - 1) // 2], "selected"] = True
    return out.sort_values(["culture", "DIV", "recording"]).reset_index(drop=True)


def _support_matrix(simulated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = []
    supports = []
    for realization in simulated:
        count, support = empirical_triplet_support_zeta(realization)
        counts.append(count)
        supports.append(support)
    return np.asarray(counts, dtype=np.int64), np.asarray(supports, dtype=float)


def _moment_rmse(model: ExactPairwise, observed: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    samples = observed.T.astype(float)
    target = np.column_stack((samples, np.column_stack(
        [samples[:, i] * samples[:, j] for i, j in model.pairs]
    ))).mean(axis=0)
    error = model.moments(probability) - target
    return float(np.sqrt(np.mean(error[:model.n] ** 2))), float(np.sqrt(np.mean(error[model.n:] ** 2)))


def evaluate_exact_arm(x: np.ndarray, theta: np.ndarray, *, n_realizations: int, seed: int) -> dict[str, object]:
    data = np.asarray(x, dtype=np.uint8)
    model = ExactPairwise(data.shape[0])
    k = data.sum(axis=0, dtype=np.int16)
    k_weights = np.bincount(k, minlength=model.n + 1).astype(float) / data.shape[1]
    probability = model.probabilities(theta, k_weights)
    simulated = model.simulate(theta, k, n_realizations, seed)
    observed_counts, observed_support = empirical_triplet_support_zeta(data)
    null_counts, null_supports = _support_matrix(simulated)
    score = triplet_residual_score(observed_support, null_supports)
    node_rmse, pair_rmse = _moment_rmse(model, data, probability)
    return {
        "simulated": simulated, "observed_counts": observed_counts,
        "observed_support": observed_support, "null_counts": null_counts,
        "null_supports": null_supports, "deterministic_support": model.triplet_support(probability),
        "deterministic_moments": model.moments(probability),
        "observed_mass": score["score"], "null_mass_mean": score["null_mean"],
        "null_mass_sd": score["null_sd"], "p_upper": score["p_upper"],
        "z": score["z"], "status": score["status"],
        "null_scores": score["null_scores"], "node_rmse": node_rmse, "pair_rmse": pair_rmse,
        "exact_k_mismatch_count": int(np.count_nonzero(simulated.sum(axis=1) != k[None, :])),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hash(model_path: Path, data_path: Path, recording: str, arm: str) -> str:
    source_dir = Path(__file__).resolve().parent
    code_dir = source_dir.parent
    payload = json.dumps({"version": PIPELINE_VERSION, "recording": recording, "arm": arm,
                          "model": _sha256(model_path), "data": _sha256(data_path),
                          "decoder_source": _sha256(code_dir / "hotopology/null_models.py"),
                          "exact_source": _sha256(source_dir / "exact_pairwise.py"),
                          "n_null": N_NULL}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def build_candidates(root: Path) -> pd.DataFrame:
    eligibility = pd.read_csv(root / "tables/computational_closure_v2/requested_node_eligibility.csv")
    rows = eligibility.loc[
        eligibility.analysis.eq("dynamic16") & eligibility.culture.isin(["P1", "P2", "P3"])
        & eligibility.DIV.isin([14, 21, 63])
    ].copy()
    means = []
    reasons = []
    for row in rows.itertuples():
        if not bool(row.eligible):
            means.append(np.nan)
            reasons.append(f"eligible_nodes={int(row.eligible_nodes)}<16")
            continue
        model_path = root / "data_intermediate/maxent_models_top16" / f"{row.recording}_maxent.mat"
        mat = loadmat(model_path, squeeze_me=True, struct_as_record=False)
        nodes = np.asarray(mat["active_nodes"], dtype=int).reshape(-1) - 1
        with np.load(root / "data_intermediate/binary_25ms" / f"{row.recording}_25ms.npz") as z:
            observed = np.asarray(z["x"], dtype=np.uint8)[nodes]
        means.append(float(observed.sum(axis=0).mean()))
        reasons.append("")
    rows["mean_population_count"] = means
    rows["reason"] = reasons
    return select_lower_median_candidates(rows)


def _finite_arm(observed: np.ndarray, fit, recording: str) -> dict[str, object]:
    n = observed.shape[0]
    h, j = decode_ising_factors(np.asarray(fit.pairwise_model.factors), n)
    k = observed.sum(axis=0, dtype=np.int16)
    seed = stable_seed(recording, "_instantaneous100_v2")
    simulated = resample_pool_conditioned_on_k(
        np.asarray(fit.pairwise_samples, dtype=np.uint8).reshape(n, -1), k,
        n_surrogates=N_NULL, seed=seed, fallback_h=h, fallback_j=j, fallback_sweeps=20)
    observed_counts, observed_support = empirical_triplet_support_zeta(observed)
    null_counts, null_supports = _support_matrix(simulated)
    score = triplet_residual_score(observed_support, null_supports)
    pooled = simulated.transpose(1, 0, 2).reshape(n, -1)
    node_rmse = float(np.sqrt(np.mean((observed.mean(axis=1) - pooled.mean(axis=1)) ** 2)))
    observed_float = observed.astype(float)
    pooled_float = pooled.astype(float)
    oi = observed_float @ observed_float.T / observed.shape[1]
    pi = pooled_float @ pooled_float.T / pooled.shape[1]
    upper = np.triu_indices(n, 1)
    pair_rmse = float(np.sqrt(np.mean((oi[upper] - pi[upper]) ** 2)))
    return {"simulated": simulated, "observed_counts": observed_counts, "observed_support": observed_support,
            "null_counts": null_counts, "null_supports": null_supports,
            "deterministic_support": null_supports.mean(axis=0),
            "deterministic_moments": np.concatenate((pooled_float.mean(axis=1), pi[upper])),
            "observed_mass": score["score"], "null_mass_mean": score["null_mean"], "null_mass_sd": score["null_sd"],
            "p_upper": score["p_upper"], "z": score["z"], "status": score["status"], "null_scores": score["null_scores"],
            "node_rmse": node_rmse, "pair_rmse": pair_rmse,
            "exact_k_mismatch_count": int(np.count_nonzero(simulated.sum(axis=1) != k[None, :])), "seed": seed}


def process_record(root: Path, output: Path, row: pd.Series) -> list[dict[str, object]]:
    recording = str(row.recording)
    started_record = time.perf_counter()
    model_path = root / "data_intermediate/maxent_models_top16" / f"{recording}_maxent.mat"
    data_path = root / "data_intermediate/binary_25ms" / f"{recording}_25ms.npz"
    mat = loadmat(model_path, squeeze_me=True, struct_as_record=False)
    fit = mat["fit"]
    nodes = np.asarray(mat["active_nodes"], dtype=int).reshape(-1) - 1
    with np.load(data_path) as z:
        observed = np.asarray(z["x"], dtype=np.uint8)[nodes]
    theta0 = theta_from_toolbox_factors(np.asarray(fit.pairwise_model.factors), observed.shape[0])
    summaries = []
    for arm in ARMS:
        started = time.perf_counter()
        ihash = _input_hash(model_path, data_path, recording, arm)
        checkpoint = output / "checkpoints" / f"{recording}_{arm}.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as old:
                if str(old["input_hash"].item()) == ihash:
                    summaries.append(json.loads(str(old["summary_json"].item())))
                    continue
        convergence = True
        message = "not_applicable"
        iterations = 0
        max_gradient = np.nan
        theta = theta0
        if arm == "finite_pool":
            result = _finite_arm(observed, fit, recording)
        else:
            if arm == "conditional_refit":
                fitted = ExactPairwise(observed.shape[0]).fit(observed, conditional=True, theta0=theta0, gtol=1e-7, maxiter=2000)
                theta = fitted["theta"]
                convergence = bool(fitted["success"])
                message = str(fitted["message"])
                iterations = int(fitted["nit"])
                max_gradient = float(fitted["max_abs_gradient"])
            seed = stable_seed(recording, "_exact" if arm == "exact" else "_refit")
            result = evaluate_exact_arm(observed, theta, n_realizations=N_NULL, seed=seed)
            result["seed"] = seed
        summary = {"recording": recording, "culture": row.culture, "DIV": int(row.DIV),
                   "n_nodes": observed.shape[0], "n_bins": observed.shape[1], "candidate_n": int(row.candidate_n),
                   **{key: result[key] for key in ("observed_mass", "null_mass_mean", "null_mass_sd", "p_upper", "z", "status", "node_rmse", "pair_rmse", "exact_k_mismatch_count", "seed")},
                   "arm": arm, "convergence": convergence, "convergence_message": message,
                   "iterations": iterations, "max_gradient": max_gradient,
                   "runtime_seconds": float(time.perf_counter() - started), "input_hash": ihash}
        centered = result["null_supports"] - result["null_supports"].mean(axis=0)
        np.savez_compressed(checkpoint, input_hash=np.array(ihash), theta=theta, theta0=theta0,
                            observed_counts=result["observed_counts"], observed_support=result["observed_support"],
                            deterministic_support=result["deterministic_support"], deterministic_moments=result["deterministic_moments"],
                            null_counts=result["null_counts"], null_supports=result["null_supports"],
                            null_loo_residuals=(N_NULL / (N_NULL - 1)) * centered,
                            null_scores=result["null_scores"], summary_json=np.array(json.dumps(summary)))
        summaries.append(summary)
    print(f"{recording}: {time.perf_counter() - started_record:.1f}s", flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("select", "pilot", "full"), default="full")
    args = parser.parse_args()
    root = args.result_dir.resolve()
    output = root / "calibration/observed_benchmark"
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(root)
    candidates.to_csv(output / "candidates.csv", index=False)
    selected = candidates.loc[candidates.selected].copy()
    if len(selected) != 9 or selected.candidate_n.tolist().count(5) != 1:
        raise RuntimeError("selection gate failed: expected nine strata and only one five-candidate stratum")
    if args.mode == "select":
        print(selected[["recording", "culture", "DIV", "candidate_n"]].to_string(index=False))
        return
    if args.mode == "pilot":
        selected = selected.iloc[:1]
    rows = [item for _, row in selected.iterrows() for item in process_record(root, output, row)]
    pd.DataFrame(rows).sort_values(["culture", "DIV", "arm"]).to_csv(output / f"summary_{args.mode}.csv", index=False)
    if any(row["exact_k_mismatch_count"] != 0 for row in rows):
        raise RuntimeError("exact-K preservation gate failed")
    bad_refits = [r["recording"] for r in rows if r["arm"] == "conditional_refit" and not r["convergence"]]
    if bad_refits:
        raise RuntimeError(f"conditional refit convergence gate failed: {bad_refits}")


if __name__ == "__main__":
    main()
