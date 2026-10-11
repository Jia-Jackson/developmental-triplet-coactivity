# Developmental triplet coactivity in mouse cortical cultures

This repository accompanies **Developmental triplet coactivity in mouse cortical cultures overlaps lower-order and temporal structure** by Xiaoli Jia, Qichao Gong, Liu Xueqing, Tahir Ali, Shupeng Li and Jinxing Feng.

The dataset contains 143 retained well recordings from 18 wells in three independent mouse cortical culture preparations (P1, P2 and P3), at DIV14, 21, 28, 35, 42, 49, 56 and 63. The native six-well plate files contain 144 candidate well recordings. WT_DIV35_P2_A1 was excluded before this analysis and remains identified in the metadata; it is not reintroduced into calculations.

## Contents

- `data/spike_events/`: lossless, gzip-compressed, two-column spike-event exports for all 143 retained recordings.
- `metadata/`: candidate/retained recording manifest, native-file hashes, figure-source index and sample-availability information.
- `source_data/`: manuscript-associated figure values. Supplementary labels are linked to their parent figures; the index also maps the previous S1–S12 labels.
- `analysis/correction/`: corrected three-preparation computational analysis and its upstream output tables.
- `analysis/upgrade/`: full-array profiles, fixed-12 projections, O-information and upstream age-prediction inputs.
- `analysis/final_revision/`: the final prediction and endpoint-z corrections. Its `corrected_*` tables take precedence over upstream prediction/endpoint tables.
- `third_party/maxent_toolbox/`: the externally developed MATLAB toolbox source and its original MIT licence, corresponding to commit 82b4528675fed43beb1d90a1c06b43189afe023e.
- `docs/`: data dictionary, computational provenance, reproduction and dependency instructions.
- `scripts/`: repository verification and safe checkpoint extraction.

## Data download

The manuscript-associated GitHub release `v1.0.0` is intended to carry eight `raw_spk_DIV*.zip` assets and `analysis_inputs_and_checkpoints.zip`. Download those assets from the repository's Releases page. Extract the analysis-input archive into the repository root with `python scripts/restore_release_asset.py PATH_TO_ARCHIVE`. Keep native SPK archives outside the Git tree or extract into the ignored `raw_spk/` folder. Native SPK files are acquisition spike-event files, not a claim that continuous voltage traces are supplied.

The CSV event exports can be read without Axion software. They retain the original event order, spike times in seconds and electrode indices 1–64. The MATLAB arrays in the checkpoint asset contain the same retained events and are the inputs to the computational pipeline. Original SPK files are unmodified and their SHA-256 hashes match the corrected input provenance.

## Reproduction and versions

Start with `docs/REPRODUCING.md`. The release preserves corrected P1/P2/P3 data and does not use the previously duplicated P3 inputs. Main null ensembles use 100 realizations. Historical 20-block proposal pools are retained only where required to reproduce the finite-pool comparator; they are not the primary 100-realization ensemble.

The upstream analysis directories preserve intermediate outputs for reproducibility. Do not cite their old age-prediction summaries as final. The manuscript's final prediction uses 27 conventional, five distinct residual/topological and 12 O-information features. Undefined endpoint z scores remain missing when the reference standard deviation is zero; native audit checkpoints can retain the original sentinel, but these values must not enter final statistics.

## Licence and citation

Author-owned analysis code and software documentation use the MIT licence in `LICENSE`. Research data and result tables use CC BY 4.0 as described in `DATA_LICENSE.md`. Third-party software retains its own licence; no new licence is imposed on third-party code or MATLAB itself. A persistent archive should cover the exact manuscript-associated release, including the large Release assets rather than only the Git source tree. `CITATION.cff` identifies the manuscript authors and release.

## Integrity

Run `python scripts/verify_repository.py`. `checksums.sha256` covers repository files, and `metadata/release_assets_manifest.csv` lists hashes and sizes of the separate Release assets. Matching hashes verifies file identity; it is not a new validation of biological conclusions.
