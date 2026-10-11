from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
import argparse
import json
import time
import zlib

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .information import (
    o_information_from_support_counts,
    o_information_many_binary,
    summarize_o_information,
    support_counts_from_binary,
)
from .io_utils import atomic_json, atomic_savez, parse_recording, sha256_file
from .projection import incidence_projection
from .vendor_kinetic_null import simulate_kinetic_conditioned_k
from .vendor_null_models import decode_ising_factors, resample_pool_conditioned_on_k


N_NULL = 100


def _load_binary(source: Path, recording: str) -> tuple[np.ndarray, dict[str, object]]:
    path = source / "data_intermediate" / "binary_25ms" / f"{recording}_25ms.npz"
    with np.load(path) as archive:
        x = np.asarray(archive["x"], dtype=np.uint8)
        metadata = {
            "recording": recording,
            "DIV": int(archive["div"]),
            "culture": f"P{int(archive['plate'])}",
            "well": str(archive["well"]),
            "n_bins": int(x.shape[1]),
        }
    metadata["subject_id"] = f"{metadata['culture']}_{metadata['well']}"
    return x, metadata


def _null_comparison(observed: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    mean = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    z = np.divide(observed - mean, sd, out=np.full_like(mean, np.nan), where=sd > 0)
    centered = np.abs(observed - mean)
    p = (1.0 + np.sum(np.abs(null - mean) >= centered, axis=0)) / (null.shape[0] + 1.0)
    return mean, sd, z, p


def process_fixed12_model(model_path_string: str, source_string: str, checkpoint_string: str) -> dict[str, object]:
    started = time.perf_counter()
    model_path = Path(model_path_string)
    source = Path(source_string)
    checkpoint = Path(checkpoint_string)
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as archive:
            return json.loads(str(archive["audit_json"]))
    recording = model_path.stem.removesuffix("_maxent")
    model = loadmat(model_path, squeeze_me=True, struct_as_record=False)
    nodes = np.asarray(model["active_nodes"], dtype=int).reshape(-1) - 1
    if len(nodes) != 12:
        raise ValueError(f"fixed12 model has {len(nodes)} nodes: {recording}")
    full, metadata = _load_binary(source, recording)
    observed = full[nodes]
    n_nodes, n_bins = observed.shape
    single_counts, pairs, pair_counts, triplets, observed_triplet_counts = support_counts_from_binary(observed)
    observed_oinfo = o_information_from_support_counts(
        single_counts, pairs, pair_counts, triplets, observed_triplet_counts, n_bins
    )

    fit = model["fit"]
    pool = np.asarray(fit.pairwise_samples, dtype=np.uint8).reshape(n_nodes, -1)
    h, j = decode_ising_factors(np.asarray(fit.pairwise_model.factors), n_nodes)
    seed = int(zlib.crc32((recording + "_instantaneous100_v2").encode("utf-8")))
    simulated = resample_pool_conditioned_on_k(
        pool,
        observed.sum(axis=0, dtype=np.int32),
        n_surrogates=N_NULL,
        seed=seed,
        fallback_h=h,
        fallback_j=j,
        fallback_sweeps=20,
    )
    expected_k = np.broadcast_to(observed.sum(axis=0), (N_NULL, n_bins))
    if not np.array_equal(simulated.sum(axis=1), expected_k):
        raise ValueError(f"instantaneous K(t) mismatch: {recording}")
    null_triplet_counts = np.empty((N_NULL, len(triplets)), dtype=np.int32)
    for realization in range(N_NULL):
        _, _, _, _, counts = support_counts_from_binary(simulated[realization], triplets)
        null_triplet_counts[realization] = counts
    null_oinfo = o_information_many_binary(simulated, triplets)
    null_mean_counts, null_sd_counts, residual_z, residual_p = _null_comparison(
        observed_triplet_counts, null_triplet_counts
    )
    residual_probability = (observed_triplet_counts - null_mean_counts) / n_bins
    null_mean_oinfo, null_sd_oinfo, oinfo_z, oinfo_p = _null_comparison(observed_oinfo, null_oinfo)

    direct_edge = incidence_projection(observed_triplet_counts / n_bins, triplets, n_nodes)
    residual_edge = incidence_projection(residual_probability, triplets, n_nodes)
    magnitude_edge = incidence_projection(np.abs(residual_probability), triplets, n_nodes)
    oinfo_edge = incidence_projection(observed_oinfo, triplets, n_nodes)
    oinfo_deviation_edge = incidence_projection(observed_oinfo - null_mean_oinfo, triplets, n_nodes)

    div, culture, well, subject_id = parse_recording(recording)
    audit = {
        "recording": recording,
        "DIV": div,
        "culture": culture,
        "well": well,
        "subject_id": subject_id,
        "n_nodes": n_nodes,
        "n_bins": n_bins,
        "n_triplets": int(len(triplets)),
        "n_null_realizations": N_NULL,
        "seed": seed,
        "exact_k_preserved": True,
        "absolute_residual_mass": float(np.mean(np.abs(residual_probability))),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    atomic_savez(
        checkpoint,
        audit_json=json.dumps(audit),
        nodes_zero_based=nodes,
        single_probability=single_counts / n_bins,
        pairs=pairs,
        pair_support_probability=pair_counts / n_bins,
        triplets=triplets,
        observed_triplet_probability=observed_triplet_counts / n_bins,
        null_mean_triplet_probability=null_mean_counts / n_bins,
        null_sd_triplet_probability=null_sd_counts / n_bins,
        residual_probability=residual_probability,
        residual_z=residual_z,
        residual_p_two_sided=residual_p,
        observed_oinfo_bits=observed_oinfo,
        null_mean_oinfo_bits=null_mean_oinfo,
        null_sd_oinfo_bits=null_sd_oinfo,
        oinfo_z=oinfo_z,
        oinfo_p_two_sided=oinfo_p,
        edges=direct_edge.edges,
        direct_triplet_edge=direct_edge.edge_values,
        direct_triplet_node=direct_edge.node_values,
        residual_edge=residual_edge.edge_values,
        residual_node=residual_edge.node_values,
        absolute_residual_edge=magnitude_edge.edge_values,
        absolute_residual_node=magnitude_edge.node_values,
        oinfo_edge=oinfo_edge.edge_values,
        oinfo_node=oinfo_edge.node_values,
        oinfo_deviation_edge=oinfo_deviation_edge.edge_values,
        oinfo_deviation_node=oinfo_deviation_edge.node_values,
    )
    return audit


def process_full_array(binary_path_string: str, checkpoint_string: str) -> dict[str, object]:
    started = time.perf_counter()
    binary_path = Path(binary_path_string)
    checkpoint = Path(checkpoint_string)
    recording = binary_path.stem.removesuffix("_25ms")
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as archive:
            return json.loads(str(archive["audit_json"]))
    with np.load(binary_path) as archive:
        full = np.asarray(archive["x"], dtype=np.uint8)
    n_nodes, n_bins = full.shape
    singles, pairs, pair_counts, triplets, triplet_counts = support_counts_from_binary(full)
    oinfo = o_information_from_support_counts(
        singles, pairs, pair_counts, triplets, triplet_counts, n_bins
    )
    direct_projection = incidence_projection(triplet_counts / n_bins, triplets, n_nodes)
    oinfo_projection = incidence_projection(oinfo, triplets, n_nodes)
    div, culture, well, subject_id = parse_recording(recording)
    audit = {
        "recording": recording,
        "DIV": div,
        "culture": culture,
        "well": well,
        "subject_id": subject_id,
        "n_nodes": n_nodes,
        "n_bins": n_bins,
        "n_pairs": int(len(pairs)),
        "n_triplets": int(len(triplets)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    atomic_savez(
        checkpoint,
        audit_json=json.dumps(audit),
        single_probability=singles / n_bins,
        pairs=pairs,
        pair_support_probability=pair_counts / n_bins,
        triplets=triplets,
        triplet_support_probability=triplet_counts / n_bins,
        oinfo_bits=oinfo,
        edges=direct_projection.edges,
        triplet_edge_projection=direct_projection.edge_values,
        triplet_node_projection=direct_projection.node_values,
        oinfo_edge_projection=oinfo_projection.edge_values,
        oinfo_node_projection=oinfo_projection.node_values,
    )
    return audit


def _summaries_matrix(oinfo: np.ndarray) -> tuple[list[str], np.ndarray]:
    summary = summarize_o_information(oinfo, axis=-1)
    names = list(summary)
    return names, np.column_stack([np.atleast_1d(summary[name]) for name in names])


def process_dynamic16_oinfo(model_path_string: str, source_string: str, checkpoint_string: str) -> dict[str, object]:
    started = time.perf_counter()
    model_path = Path(model_path_string)
    source = Path(source_string)
    checkpoint = Path(checkpoint_string)
    recording = model_path.stem.removesuffix("_maxent")
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as archive:
            return json.loads(str(archive["audit_json"]))
    model = loadmat(model_path, squeeze_me=True, struct_as_record=False)
    nodes = np.asarray(model["active_nodes"], dtype=int).reshape(-1) - 1
    if len(nodes) != 16:
        raise ValueError(f"dynamic model has {len(nodes)} nodes: {recording}")
    full, _ = _load_binary(source, recording)
    observed = full[nodes]
    n_nodes, n_bins = observed.shape
    singles, pairs, pair_counts, triplets, triplet_counts = support_counts_from_binary(observed)
    observed_oinfo = o_information_from_support_counts(
        singles, pairs, pair_counts, triplets, triplet_counts, n_bins
    )

    fit = model["fit"]
    pool = np.asarray(fit.pairwise_samples, dtype=np.uint8).reshape(n_nodes, -1)
    h, j = decode_ising_factors(np.asarray(fit.pairwise_model.factors), n_nodes)
    instantaneous_seed = int(zlib.crc32((recording + "_instantaneous100_v2").encode("utf-8")))
    instantaneous = resample_pool_conditioned_on_k(
        pool,
        observed.sum(axis=0, dtype=np.int32),
        n_surrogates=N_NULL,
        seed=instantaneous_seed,
        fallback_h=h,
        fallback_j=j,
        fallback_sweeps=20,
    )
    k_sequence = observed.sum(axis=0, dtype=np.int32)
    if not np.array_equal(instantaneous.sum(axis=1), np.broadcast_to(k_sequence, (N_NULL, n_bins))):
        raise ValueError(f"instantaneous K(t) mismatch: {recording}")
    instantaneous_oinfo = o_information_many_binary(instantaneous, triplets)
    del instantaneous

    kinetic_path = source / "data_intermediate" / "kinetic_models_top16" / f"{recording}_kinetic_fit.npz"
    with np.load(kinetic_path) as kinetic_fit:
        kinetic_nodes = np.asarray(kinetic_fit["active_nodes_zero_based"], dtype=int)
        intercepts = np.asarray(kinetic_fit["intercepts"], dtype=float)
        weights = np.asarray(kinetic_fit["weights"], dtype=float)
    if not np.array_equal(nodes, kinetic_nodes):
        raise ValueError(f"kinetic/instantaneous node mismatch: {recording}")
    kinetic_seed = int(zlib.crc32((recording + "_kinetic_v2").encode("utf-8")))
    kinetic = simulate_kinetic_conditioned_k(
        intercepts,
        weights,
        k_sequence,
        initial_state=observed[:, 0],
        n_realizations=N_NULL,
        seed=kinetic_seed,
    )
    if not np.array_equal(kinetic.sum(axis=1), np.broadcast_to(k_sequence, (N_NULL, n_bins))):
        raise ValueError(f"kinetic K(t) mismatch: {recording}")
    kinetic_oinfo = o_information_many_binary(kinetic, triplets)
    del kinetic

    names, observed_summary = _summaries_matrix(observed_oinfo)
    _, instantaneous_summary = _summaries_matrix(instantaneous_oinfo)
    _, kinetic_summary = _summaries_matrix(kinetic_oinfo)
    div, culture, well, subject_id = parse_recording(recording)
    audit = {
        "recording": recording,
        "DIV": div,
        "culture": culture,
        "well": well,
        "subject_id": subject_id,
        "n_nodes": n_nodes,
        "n_bins": n_bins,
        "n_triplets": int(len(triplets)),
        "n_null_realizations_each": N_NULL,
        "instantaneous_seed": instantaneous_seed,
        "kinetic_seed": kinetic_seed,
        "instantaneous_exact_k_preserved": True,
        "kinetic_exact_k_preserved": True,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    atomic_savez(
        checkpoint,
        audit_json=json.dumps(audit),
        nodes_zero_based=nodes,
        triplets=triplets,
        metric_names=np.asarray(names),
        observed_summary=observed_summary.reshape(-1),
        instantaneous_null_summary=instantaneous_summary,
        kinetic_null_summary=kinetic_summary,
        observed_oinfo_bits=observed_oinfo,
        instantaneous_null_mean_oinfo_bits=instantaneous_oinfo.mean(axis=0),
        kinetic_null_mean_oinfo_bits=kinetic_oinfo.mean(axis=0),
    )
    return audit


def _run_parallel(jobs: list[tuple], function, max_workers: int, label: str) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(function, *job) for job in jobs]
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if number % 5 == 0 or number == len(futures):
                print(f"{label}: {number}/{len(futures)} complete", flush=True)
    return sorted(results, key=lambda item: str(item["recording"]))


def _consolidate_fixed12(checkpoints: Path, table_dir: Path, source: Path) -> None:
    reference = pd.read_csv(
        source / "tables/computational_closure_v2/v2_instantaneous_null100_fixed_n12_triplet_residuals.csv"
    ).set_index("recording")
    triplet_rows, edge_rows, node_rows, audits = [], [], [], []
    for path in sorted(checkpoints.glob("*.npz")):
        with np.load(path, allow_pickle=False) as a:
            audit = json.loads(str(a["audit_json"]))
            audits.append(audit)
            recording = audit["recording"]
            expected = float(reference.loc[recording, "absolute_residual_mass"])
            if not np.isclose(audit["absolute_residual_mass"], expected, rtol=1e-10, atol=1e-15):
                raise AssertionError(f"fixed12 replay audit failed: {recording}")
            nodes = a["nodes_zero_based"]
            triplets = a["triplets"]
            observed_triplet = a["observed_triplet_probability"]
            null_mean_triplet = a["null_mean_triplet_probability"]
            residual = a["residual_probability"]
            residual_z = a["residual_z"]
            residual_p = a["residual_p_two_sided"]
            observed_oinfo = a["observed_oinfo_bits"]
            null_mean_oinfo = a["null_mean_oinfo_bits"]
            for index, (i, j, k) in enumerate(triplets):
                triplet_rows.append({
                    **{key: audit[key] for key in ("recording", "DIV", "culture", "well", "subject_id")},
                    "local_i": int(i), "local_j": int(j), "local_k": int(k),
                    "electrode_i": int(nodes[i]) + 1, "electrode_j": int(nodes[j]) + 1,
                    "electrode_k": int(nodes[k]) + 1,
                    "observed_support_probability": float(observed_triplet[index]),
                    "null_mean_support_probability": float(null_mean_triplet[index]),
                    "residual_probability": float(residual[index]),
                    "absolute_residual_probability": float(abs(residual[index])),
                    "residual_z": float(residual_z[index]),
                    "residual_p_two_sided": float(residual_p[index]),
                    "observed_oinfo_bits": float(observed_oinfo[index]),
                    "null_mean_oinfo_bits": float(null_mean_oinfo[index]),
                    "oinfo_deviation_bits": float(observed_oinfo[index] - null_mean_oinfo[index]),
                })
            edges = a["edges"]
            pair_support = a["pair_support_probability"]
            direct_edge = a["direct_triplet_edge"]
            residual_edge_values = a["residual_edge"]
            absolute_residual_edge = a["absolute_residual_edge"]
            oinfo_edge_values = a["oinfo_edge"]
            oinfo_deviation_edge_values = a["oinfo_deviation_edge"]
            for index, (i, j) in enumerate(edges):
                edge_rows.append({
                    **{key: audit[key] for key in ("recording", "DIV", "culture", "well", "subject_id")},
                    "local_i": int(i), "local_j": int(j), "electrode_i": int(nodes[i]) + 1,
                    "electrode_j": int(nodes[j]) + 1,
                    "pair_support_probability": float(pair_support[index]),
                    "triplet_support_projection": float(direct_edge[index]),
                    "residual_projection": float(residual_edge_values[index]),
                    "absolute_residual_projection": float(absolute_residual_edge[index]),
                    "oinfo_projection_bits": float(oinfo_edge_values[index]),
                    "oinfo_deviation_projection_bits": float(oinfo_deviation_edge_values[index]),
                })
            single_probability = a["single_probability"]
            direct_node = a["direct_triplet_node"]
            residual_node_values = a["residual_node"]
            absolute_residual_node = a["absolute_residual_node"]
            oinfo_node_values = a["oinfo_node"]
            oinfo_deviation_node_values = a["oinfo_deviation_node"]
            for local, electrode in enumerate(nodes):
                node_rows.append({
                    **{key: audit[key] for key in ("recording", "DIV", "culture", "well", "subject_id")},
                    "local_node": local, "electrode": int(electrode) + 1,
                    "firing_probability": float(single_probability[local]),
                    "triplet_support_projection": float(direct_node[local]),
                    "residual_projection": float(residual_node_values[local]),
                    "absolute_residual_projection": float(absolute_residual_node[local]),
                    "oinfo_projection_bits": float(oinfo_node_values[local]),
                    "oinfo_deviation_projection_bits": float(oinfo_deviation_node_values[local]),
                })
    pd.DataFrame(audits).to_csv(table_dir / "v2_fixed12_full_profile_audit.csv", index=False)
    pd.DataFrame(triplet_rows).to_csv(table_dir / "v2_fixed12_triplet_profiles.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(table_dir / "v2_fixed12_edge_projections.csv", index=False)
    pd.DataFrame(node_rows).to_csv(table_dir / "v2_fixed12_node_projections.csv", index=False)


def _consolidate_full_array(checkpoints: Path, table_dir: Path) -> None:
    edge_rows, node_rows, audits = [], [], []
    for path in sorted(checkpoints.glob("*.npz")):
        with np.load(path, allow_pickle=False) as a:
            audit = json.loads(str(a["audit_json"]))
            audits.append(audit)
            metadata = {key: audit[key] for key in ("recording", "DIV", "culture", "well", "subject_id")}
            edges = a["edges"]
            pair_support = a["pair_support_probability"]
            triplet_edge = a["triplet_edge_projection"]
            oinfo_edge = a["oinfo_edge_projection"]
            for index, (i, j) in enumerate(edges):
                edge_rows.append({
                    **metadata, "electrode_i": int(i) + 1, "electrode_j": int(j) + 1,
                    "pair_support_probability": float(pair_support[index]),
                    "triplet_support_projection": float(triplet_edge[index]),
                    "oinfo_projection_bits": float(oinfo_edge[index]),
                })
            single_probability = a["single_probability"]
            triplet_node = a["triplet_node_projection"]
            oinfo_node = a["oinfo_node_projection"]
            for node in range(len(single_probability)):
                node_rows.append({
                    **metadata, "electrode": node + 1,
                    "firing_probability": float(single_probability[node]),
                    "triplet_support_projection": float(triplet_node[node]),
                    "oinfo_projection_bits": float(oinfo_node[node]),
                })
    pd.DataFrame(audits).to_csv(table_dir / "v2_full_array_profile_audit.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(table_dir / "v2_full_array_edge_projections.csv", index=False)
    pd.DataFrame(node_rows).to_csv(table_dir / "v2_full_array_node_projections.csv", index=False)


def _consolidate_oinfo(checkpoints: Path, table_dir: Path) -> None:
    comparison_rows, triplet_rows, audits = [], [], []
    for path in sorted(checkpoints.glob("*.npz")):
        with np.load(path, allow_pickle=False) as a:
            audit = json.loads(str(a["audit_json"]))
            audits.append(audit)
            metadata = {key: audit[key] for key in ("recording", "DIV", "culture", "well", "subject_id")}
            names = [str(name) for name in a["metric_names"]]
            observed = a["observed_summary"]
            instantaneous_summary = a["instantaneous_null_summary"]
            kinetic_summary = a["kinetic_null_summary"]
            for model, null in (
                ("instantaneous_count_conditioned", instantaneous_summary),
                ("kinetic_count_conditioned", kinetic_summary),
            ):
                for index, metric in enumerate(names):
                    mean, sd, z, p = _null_comparison(np.asarray([observed[index]]), null[:, [index]])
                    comparison_rows.append({
                        **metadata, "model": model, "metric": metric,
                        "observed": float(observed[index]), "null_mean": float(mean[0]),
                        "null_sd": float(sd[0]), "z": float(z[0]),
                        "empirical_p_two_sided": float(p[0]), "n_null_realizations": N_NULL,
                    })
            nodes = a["nodes_zero_based"]
            triplets = a["triplets"]
            observed_oinfo = a["observed_oinfo_bits"]
            instantaneous_mean = a["instantaneous_null_mean_oinfo_bits"]
            kinetic_mean = a["kinetic_null_mean_oinfo_bits"]
            for index, (i, j, k) in enumerate(triplets):
                triplet_rows.append({
                    **metadata, "electrode_i": int(nodes[i]) + 1, "electrode_j": int(nodes[j]) + 1,
                    "electrode_k": int(nodes[k]) + 1,
                    "observed_oinfo_bits": float(observed_oinfo[index]),
                    "instantaneous_null_mean_oinfo_bits": float(instantaneous_mean[index]),
                    "kinetic_null_mean_oinfo_bits": float(kinetic_mean[index]),
                })
    pd.DataFrame(audits).to_csv(table_dir / "v2_dynamic16_oinfo_audit.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(table_dir / "v2_dynamic16_oinfo_null_comparisons.csv", index=False)
    pd.DataFrame(triplet_rows).to_csv(table_dir / "v2_dynamic16_oinfo_triplets.csv", index=False)


def build_input_manifest(source: Path, output: Path) -> None:
    patterns = [
        "data_intermediate/binary_25ms/*.npz",
        "data_intermediate/maxent_models_fixed_n12/*.mat",
        "data_intermediate/maxent_models_top16/*.mat",
        "data_intermediate/kinetic_models_top16/*.npz",
        "tables/computational_closure_v2/v2_recording_key_manifest.csv",
        "tables/computational_closure_v2/v2_fixed_node_manifest_long.csv",
        "tables/computational_closure_v2/v2_instantaneous_null100_fixed_n12_triplet_residuals.csv",
        "data_intermediate/baseline_bridge/WT_development_original27_corrected_filtered_preimputation.csv",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(source.glob(pattern))
    rows = []
    for path in sorted(set(files)):
        rows.append({
            "relative_path": str(path.relative_to(source)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    atomic_json(output / "audit" / "v2_input_hash_manifest.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["manifest", "full", "fixed12", "oinfo", "all"], default="all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    table_dir = output / "tables" / "upgrade_v2"
    table_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in {"manifest", "all"}:
        build_input_manifest(source, output)
    if args.phase in {"full", "all"}:
        checkpoint_dir = output / "intermediate" / "full_array_profiles"
        jobs = []
        for path in sorted((source / "data_intermediate/binary_25ms").glob("*_25ms.npz")):
            recording = path.stem.removesuffix("_25ms")
            jobs.append((str(path), str(checkpoint_dir / f"{recording}.npz")))
        results = _run_parallel(jobs, process_full_array, args.workers, "full-array profiles")
        atomic_json(output / "logs" / "full_array_compute.json", results)
        _consolidate_full_array(checkpoint_dir, table_dir)
    if args.phase in {"fixed12", "all"}:
        checkpoint_dir = output / "intermediate" / "fixed12_profiles"
        jobs = []
        for path in sorted((source / "data_intermediate/maxent_models_fixed_n12").glob("*_maxent.mat")):
            recording = path.stem.removesuffix("_maxent")
            jobs.append((str(path), str(source), str(checkpoint_dir / f"{recording}.npz")))
        results = _run_parallel(jobs, process_fixed12_model, args.workers, "fixed12 null profiles")
        atomic_json(output / "logs" / "fixed12_compute.json", results)
        _consolidate_fixed12(checkpoint_dir, table_dir, source)
    if args.phase in {"oinfo", "all"}:
        checkpoint_dir = output / "intermediate" / "dynamic16_oinfo"
        jobs = []
        for path in sorted((source / "data_intermediate/maxent_models_top16").glob("*_maxent.mat")):
            recording = path.stem.removesuffix("_maxent")
            jobs.append((str(path), str(source), str(checkpoint_dir / f"{recording}.npz")))
        results = _run_parallel(jobs, process_dynamic16_oinfo, args.workers, "dynamic16 O-information")
        atomic_json(output / "logs" / "dynamic16_oinfo_compute.json", results)
        _consolidate_oinfo(checkpoint_dir, table_dir)


if __name__ == "__main__":
    main()
