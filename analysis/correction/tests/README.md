# Active correction test suite

Run `python -m pytest` from the corrected project with `PYTHONPATH=code`.

`tests/legacy` is the frozen copied original test snapshot. Its old artifact-layout tests target the former 10-main-figure/story-v3 output trees and derive their project root from their former location. Recursively collecting that snapshot in its archive location on 2026-09-12 produced 10 missing-path/old-layout failures and 126 passes; these were not scientific-calculation failures. The active suite does not recursively collect that historical fixture layout.

`test_corrected_figure_outputs.py` checks the current six-main/ten-supplement layout, annotation values, current 100-realization source, and undefined endpoint sample sizes. Numerical correction tests cover input identity, baseline mapping, node eligibility, finite-state calculations, score calibration, native factor sign, caching, bootstrap inputs, and synthetic failure handling. The original snapshot is retained unchanged for audit; it is not claimed to pass as a standalone relocated project.
