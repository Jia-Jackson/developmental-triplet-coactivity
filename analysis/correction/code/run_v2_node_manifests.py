from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from hotopology.metrics import compute_direct_metrics
from hotopology.node_selection import rank_fixed_longitudinal_nodes, select_fixed_node_sets
from requested_node_eligibility import fixed_sets


PATTERN = re.compile(r"WT_DIV(?P<DIV>\d+)_P(?P<plate>\d+)_(?P<well>[AB]\d)_25ms\.npz")
NODE_COUNTS = [12, 16, 20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    binary_dir = root / "data_intermediate" / "binary_25ms"
    out = root / "tables" / "computational_closure_v2"
    out.mkdir(parents=True, exist_ok=True)

    records = []
    by_subject: dict[str, dict[int, tuple[str, np.ndarray]]] = {}
    for path in sorted(binary_dir.glob("*_25ms.npz")):
        match = PATTERN.fullmatch(path.name)
        if not match:
            continue
        div = int(match.group("DIV"))
        plate = int(match.group("plate"))
        well = match.group("well")
        subject = f"P{plate}_{well}"
        recording = path.stem.removesuffix("_25ms")
        with np.load(path) as archive:
            x = np.asarray(archive["x"], dtype=np.uint8)
        by_subject.setdefault(subject, {})[div] = (recording, x)

    manifest_rows = []
    ranking_rows = []
    direct_rows = []
    subject_summary = []
    for subject, age_map in sorted(by_subject.items()):
        arrays = {div: item[1] for div, item in age_map.items()}
        ranking = rank_fixed_longitudinal_nodes(
            arrays, min_active_bins=20, min_inactive_bins=20
        )
        plate_text, well = subject.split("_", 1)
        plate = int(plate_text.removeprefix("P"))
        ranking.insert(0, "subject_id", subject)
        ranking.insert(1, "culture", plate_text)
        ranking.insert(2, "plate", plate)
        ranking.insert(3, "well", well)
        ranking_rows.append(ranking)
        selected, _ = fixed_sets(arrays, NODE_COUNTS)
        subject_summary.append(
            {
                "subject_id": subject,
                "culture": plate_text,
                "well": well,
                "n_available_ages": len(age_map),
                "n_common_eligible_nodes": len(ranking),
                "minimum_selected_n": min(selected) if selected else None,
                "maximum_selected_n": max(selected) if selected else None,
            }
        )
        for n_nodes, nodes in selected.items():
            for rank, node in enumerate(nodes, start=1):
                for div, (recording, _) in sorted(age_map.items()):
                    manifest_rows.append(
                        {
                            "recording": recording,
                            "DIV": div,
                            "plate": plate,
                            "culture": plate_text,
                            "well": well,
                            "subject_id": subject,
                            "n_nodes": n_nodes,
                            "fixed_rank": rank,
                            "electrode_index_zero_based": int(node),
                            "electrode_index_one_based": int(node + 1),
                        }
                    )
            for div, (recording, x) in sorted(age_map.items()):
                metrics, _ = compute_direct_metrics(x[nodes], min_active_bins=1)
                direct_rows.append(
                    {
                        "recording": recording,
                        "DIV": div,
                        "plate": plate,
                        "culture": plate_text,
                        "well": well,
                        "subject_id": subject,
                        "node_selection": "fixed_longitudinal_maximin",
                        "n_nodes_requested": n_nodes,
                        **metrics,
                    }
                )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["n_nodes", "DIV", "plate", "well", "fixed_rank"]
    )
    ranking_table = pd.concat(ranking_rows, ignore_index=True)
    direct = pd.DataFrame(direct_rows).sort_values(
        ["n_nodes_requested", "DIV", "plate", "well"]
    )
    manifest.to_csv(out / "v2_fixed_node_manifest_long.csv", index=False)
    ranking_table.to_csv(out / "v2_fixed_node_rankings.csv", index=False)
    direct.to_csv(out / "v2_fixed_node_direct_metrics_25ms.csv", index=False)
    pd.DataFrame(subject_summary).to_csv(
        out / "v2_fixed_node_subject_summary.csv", index=False
    )
    audit = {
        "n_subjects": len(by_subject),
        "n_recordings": sum(len(value) for value in by_subject.values()),
        "node_counts": NODE_COUNTS,
        "minimum_common_eligible_nodes": int(
            min(item["n_common_eligible_nodes"] for item in subject_summary)
        ),
        "nested_sets": True,
        "n_direct_rows": len(direct),
    }
    (root / "qa" / "v2_fixed_node_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
