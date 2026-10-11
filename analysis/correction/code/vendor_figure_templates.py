"""Mechanically relocate author-approved Python panel templates into this correction.

No original file is modified; every copied/transformed template is hash-audited.
This script does not execute plotting or copy old numerical figure outputs.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OLD = Path(r"D:\Published paper\high-order-topology\Data_processing\Result")
FIG = Path(r"D:\Published paper\high-order-topology\CSF_prepartion\Figure")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    sources = [p for p in FIG.rglob("*.py") if p.parent.name in
               {*(f"Figure.{i}" for i in range(2, 7)), *(f"Figure.S{i}" for i in range(2, 11))}
               and not p.name.startswith("test_")]
    builder = OLD / "Chaos, Solitons & Fractals/code/build_csf_figures.py"
    rows = []
    for source in [builder, *sorted(sources)]:
        raw = source.read_bytes()
        text = raw.decode("utf-8-sig")
        if source == builder:
            dest = ROOT / "figure_bundle/code/build_csf_figures.py"
        else:
            dest = ROOT / "figures/individual" / source.relative_to(FIG)
        text = text.replace(str(OLD), str(ROOT)).replace(str(FIG), str(ROOT / "figures/individual"))
        text = text.replace("Chaos, Solitons & Fractals", "figure_bundle")
        # Only these two templates use dynamic16 null rows; full-array guards stay143.
        if source.name in {"plot_figure_4b_4c_dual_statistics.py", "plot_figure_5_panels_dual_statistics.py"}:
            text = text.replace("len(data) != 143", "len(data) != 142").replace("Expected 143 recording rows", "Expected 142 eligible recording rows")
        # Actual equal halves range149.875–150s; avoid claiming every half is150s.
        if source.name == "plot_figure_s8_revised.py":
            text = text.replace('"150-s halves"', '"Equal-duration halves"')
        if str(OLD) in text or str(FIG) in text:
            raise ValueError(f"Unresolved old path: {source}")
        dest.resolve().relative_to(ROOT)
        if dest.exists():
            raise FileExistsError(f"Refuse to overwrite an adapted template: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        rows.append({"source": str(source), "output": str(dest), "source_sha256": sha(raw),
                     "output_sha256": sha(dest.read_bytes()), "numerical_outputs_copied": False})
    audit = ROOT / "logs/figure_template_manifest.json"
    audit.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"templates": len(rows), "audit": str(audit)}))

if __name__ == "__main__":
    main()
