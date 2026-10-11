# Reproducing the computational analysis

Run commands from the repository root. Use Python 3.12 and install the dependencies listed in `requirements.txt`. `docs/observed_science_environment.txt` records the versions used for the original calculations. New Python/package versions can change optimization or numerical results; the archived inputs and output tables remain the comparison targets.

On Windows PowerShell, expose the repository root before running the copied scripts:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
```

On Linux/macOS, set `PYTHONPATH` to the absolute repository-root directory. Run calculations in a separate working copy, because the original stage scripts can rewrite generated tables in their analysis directory.

## Inspect the data without rerunning models

```text
python scripts/verify_repository.py
```

Read the 143 `.csv.gz` event files with pandas or a standard gzip/CSV reader. Compare figure values in `source_data/` and the authoritative final correction tables in `analysis/final_revision/tables/`.

## Restore archived model inputs

Download `analysis_inputs_and_checkpoints.zip` from Release v1.0.0. Then run:

```text
python scripts/restore_release_asset.py /path/to/analysis_inputs_and_checkpoints.zip
```

This restores MATLAB spike arrays, 25-ms matrices, fitted instantaneous and kinetic models, direct/null checkpoints, profile/O-information arrays, and calibration inputs in their original relative layouts. Existing files with different bytes are never overwritten. The archive is intended for the repository root, not a directory named after the ZIP file.

## Direct measurements and V2 summaries

```text
python analysis/correction/code/run_direct_analysis.py --result-dir analysis/correction --workers 2
```

That command can reuse archived direct checkpoints. For a fresh recomputation, copy the project to a separate working directory and remove only the generated direct checkpoints there; do not delete native data or reference outputs.

For the upgrade, expose the analysis packages and execute from the upgrade code directory:

```text
cd analysis/upgrade/code
python -m higher_order_v2.compute --source ../../correction --output .. --phase all --workers 2
python -m higher_order_v2.analyze_profiles --source ../../correction --output .. --phase all
python -m higher_order_v2.analyze_oinfo --source ../../correction --output ..
cd ../../..
```

The top-level repository should be on PYTHONPATH for any historical script containing `from release_paths import repository_path`. Setting this changes path resolution only, not the analytical formulas. Do not use the original upstream prediction command as the final manuscript result.

## Final corrected prediction

```text
python analysis/final_revision/code/correct_prediction_analysis.py --source analysis/correction --v2-root analysis/upgrade --source-code analysis/correction/code --v2-code analysis/upgrade/code --output reproduced/final_revision
```

The input contains 142 eligible records. The final procedure excludes the duplicated positive residual-z column and mathematically undefined endpoint-count predictor, uses fold-local imputation/scaling and reproduces the 27/5/12 feature comparison. It also exports corrected endpoint-z summaries. The published target tables are under `analysis/final_revision/tables/`; the command writes to a separate `reproduced/` directory.

## Fit models from the binary inputs

Native MATLAB and the maximum-entropy toolbox are required only for refitting the instantaneous models. They are not required to inspect the published events, tables or already fitted models. Obtain/compile the native MEX functions for the recorded toolbox version; the prepared repository includes toolbox source, not proprietary MATLAB or platform-specific executable binaries. In a copied working analysis directory, add `analysis/correction/code` to the MATLAB path and call `run_maxent_all` with the working directory, toolbox path, a 16-node request and the frozen requested-node eligibility manifest. Fixed-node models additionally require `v2_fixed_node_manifest_long.csv`. The exact signatures and fitting options remain in the archived MATLAB code.

The scripts `run_v2_instantaneous_null.py`, `run_v2_kinetic_null.py`, `run_v2_node_robustness.py`, `run_v2_hierarchical_statistics.py`, `run_v2_activity_adjustment.py`, `run_v2_stability.py` and `run_v2_extended_statistics.py` expose their analysis-directory parameters via `--help`. Use the original frozen node manifests, 100-realization settings and recorded seeds. Full refitting/simulation is computationally expensive and was not rerun during release preparation.

## Calibration

`analysis/correction/code/calibration/` contains the exact, conditional-refit and synthetic benchmark implementation. The restored `calibration/` directory supplies the original fixed design, stage-specific seeds, observed prototypes and synthetic outcomes. `run_synthetic_calibration.py` accepts explicit `--matlab` and `--toolbox` paths. Administrative bootstrap, old paths to earlier projects and legacy compatibility analyses are historical provenance, not prerequisites for rerunning from the exported inputs.

## Scope of release verification

The release-preparation checks cover file identity, raw/event cohort closure, native-to-exported event equality, package size and manuscript edit preservation. Unit tests cover the small analysis functions. They do not constitute a second full model fit, a new statistical analysis or a validation of closed-loop neuronal performance.

To verify the analysis helper functions after installing dependencies, run `python -m pytest analysis/upgrade/tests -q -p no:cacheprovider` from the repository root. Fourteen tests cover information identities, projection, longitudinal profile helpers, feature grouping and fixed-node contracts. They use small constructed examples rather than rerunning all MEA recordings.

After editing repository metadata or adding the confirmed licences, run `python scripts/write_checksums.py` to update the repository file manifest, then rerun the integrity check.
