"""Fail-fast audit of the completed higher-order topology result package."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


EXPECTED_DIV_COUNTS = {14: 18, 21: 18, 28: 18, 35: 17, 42: 18, 49: 18, 56: 18, 63: 18}


def load_csv(result_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(result_dir / "tables" / name)


def audit_docx(docx_path: Path) -> dict[str, int | str]:
    with zipfile.ZipFile(docx_path) as archive:
        assert archive.testzip() is None, "DOCX ZIP integrity failed"
        names = archive.namelist()
        document = ET.fromstring(archive.read("word/document.xml"))
        relationships = ET.fromstring(archive.read("word/_rels/document.xml.rels"))

    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
        "r": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    equation_count = len(document.findall(".//m:oMath", ns))
    media_count = len([name for name in names if name.startswith("word/media/")])
    page_size = document.find(".//w:sectPr/w:pgSz", ns)
    assert page_size is not None
    width = int(page_size.attrib[f"{{{ns['w']}}}w"])
    height = int(page_size.attrib[f"{{{ns['w']}}}h"])
    targets = [element.attrib.get("Target", "") for element in relationships]

    assert equation_count >= 8, equation_count
    assert media_count == 5, media_count
    assert (width, height) == (12240, 15840), (width, height)
    assert all("NULL" not in target for target in targets)
    return {
        "docx_equations": equation_count,
        "docx_embedded_media": media_count,
        "docx_page_size_twips": f"{width}x{height}",
    }


def verify(result_dir: Path) -> dict[str, object]:
    metadata = pd.read_csv(result_dir / "data_intermediate" / "metadata" / "recording_metadata.csv")
    manifest = pd.read_csv(result_dir / "data_intermediate" / "metadata" / "recording_manifest_all.csv")
    direct = load_csv(result_dir, "direct_higher_order_metrics_all_bins.csv")
    burst = load_csv(result_dir, "burst_stratified_higher_order_metrics_25ms.csv")
    qc = load_csv(result_dir, "maxent_model_qc.csv")
    topology = load_csv(result_dir, "topology_null_comparisons.csv")
    irreducible = load_csv(result_dir, "irreducible_triplet_summary.csv")
    exploratory = load_csv(result_dir, "exploratory_top_triplet_residuals.csv")

    assert len(metadata) == 143
    assert metadata["recording"].nunique() == 143
    assert metadata["duration_s_meta"].eq(300).all()
    assert metadata["fs_meta"].eq(12_500).all()
    assert len(manifest) == 144
    assert manifest["excluded"].sum() == 1
    assert len(list((result_dir / "data_intermediate" / "spike_arrays").glob("*.mat"))) == 143
    assert len(direct) == 572
    assert direct["recording"].nunique() == 143
    assert set(direct["bin_ms"].unique()) == {10, 25, 50, 100}
    primary = direct.loc[direct["bin_ms"] == 25]
    div_counts = primary.groupby("DIV")["recording"].nunique().to_dict()
    assert div_counts == EXPECTED_DIV_COUNTS, div_counts
    assert len(burst) == 286
    assert set(burst["stratum"].unique()) == {"inside_burst", "outside_burst"}
    assert len(qc) == 143
    assert qc["recording"].nunique() == 143
    assert qc["pairwise_solver_converged"].all()
    assert qc["model_qc_pass"].all()
    assert qc["pairwise_firing_rmse"].max() <= 0.02
    assert qc["pairwise_pair_rmse"].max() <= 0.02
    assert len(topology) == 2574
    assert set(topology["model"].unique()) == {"independent", "pairwise", "pairwise_conditioned_k"}
    assert topology.groupby(["model", "metric"]).size().eq(143).all()
    assert len(irreducible) == 286
    assert irreducible.groupby("model").size().to_dict() == {"pairwise": 143, "pairwise_conditioned_k": 143}
    assert len(exploratory) == 57_200
    assert exploratory.groupby("recording").size().eq(400).all()

    for log_name in ("direct_analysis_failures.json", "null_analysis_failures.json"):
        failures = json.loads((result_dir / "logs" / log_name).read_text(encoding="utf-8"))
        assert failures == [], (log_name, failures)

    checkpoints = list((result_dir / "data_intermediate" / "null_checkpoints").glob("*.json"))
    assert len(checkpoints) == 143
    versions = Counter(json.loads(path.read_text(encoding="utf-8"))["pipeline_version"] for path in checkpoints)
    assert versions == Counter({4: 143}), versions

    figure_counts = Counter(path.suffix.lower() for path in (result_dir / "figures").iterdir() if path.is_file())
    assert figure_counts == Counter({".svg": 5, ".pdf": 5, ".png": 5, ".tiff": 5}), figure_counts
    assert len(list((result_dir / "source_data").glob("*.csv"))) == 7
    rendered_pages = list((result_dir / "qa" / "report_render_v3").glob("page-*.png"))
    assert len(rendered_pages) == 13

    report = audit_docx(result_dir / "Higher_order_topology_analysis_report.docx")
    report.update(
        {
            "retained_recordings": int(primary["recording"].nunique()),
            "manifest_recordings": len(manifest),
            "excluded_recordings": int(manifest["excluded"].sum()),
            "converted_spike_arrays": len(list((result_dir / "data_intermediate" / "spike_arrays").glob("*.mat"))),
            "direct_metric_rows": len(direct),
            "burst_stratified_rows": len(burst),
            "maxent_qc_rows": len(qc),
            "max_firing_rmse": float(qc["pairwise_firing_rmse"].max()),
            "max_pairwise_rmse": float(qc["pairwise_pair_rmse"].max()),
            "topology_null_rows": len(topology),
            "irreducible_summary_rows": len(irreducible),
            "exploratory_triplet_rows": len(exploratory),
            "checkpoint_versions": dict(versions),
            "figure_files_by_extension": dict(figure_counts),
            "rendered_report_pages": len(rendered_pages),
            "failure_logs_empty": True,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.result_dir), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
