# Data dictionary

## Recording manifest

`recording` identifies one age, preparation and well: WT_DIV14_P1_A1. `culture` is the independent preparation P1/P2/P3. `well` is A1/A2/A3/B1/B2/B3, and `subject_id` combines preparation and well for longitudinal pairing. Wells, electrodes, triplets and model realizations are not independent culture preparations.

`DIV` is days in vitro; `duration_s` is the measured recording duration in seconds; `sampling_frequency_hz` is the acquisition sampling frequency. Nominal recordings are 300 s, whereas 12 P3 recordings at DIV28 and DIV63 are 299.75 s. No padding is used. `excluded` identifies the prespecified recording-level exclusion, and `exclusion_reason` documents its scope. `n_spikes` counts exported spike events.

`event_file` is relative to the repository root. `raw_file` is the original plate-level acquisition file inside the corresponding raw Release archive. SHA-256 fields identify original files or canonical event arrays; a native-file hash and an event-array hash are not interchangeable.

## Event files

Each `.csv.gz` has `spike_time_s` and `electrode_index_1based`. Times are stored with 17 significant digits to preserve IEEE float64 values on round-trip. Electrode index is an integer from 1 to 64. Convert to zero-based indices only at the analysis boundary. There is no amplitude or neuron-identity column; an electrode can record multiple neurons.

## Available samples

The whole-array analysis retains 143 recordings; dynamic-16 analyses have 142 eligible recordings. Fixed-12, fixed-16 and fixed-20 specifications have 143, 135 and 127 eligible recordings respectively. Individual metrics can have fewer finite observations. Use the metric-specific sample-size tables, not a uniform n. Missing and mathematically undefined scores must not be replaced with zeros or extreme finite values. Seventeen wells have all eight retained ages.

## Results and statistics

Figure files and result CSVs retain their original explicit column names. `model` distinguishes instantaneous count-conditioned pairwise and lag-1 kinetic references. `metric`, `source_metric`, `analysis`, `DIV`, `culture`, `subject_id`, and `recording` identify the applicable measurement and sampling specification. Confidence intervals, raw P values and multiplicity-adjusted P values must be distinguished by their exported column names. Model-relative z scores are standardized deviations, not measures of biological energy or causal interactions.

Blank CSV cells and NaN denote unavailable measurements. Infinite persistence intervals can be summarized with an explicitly capped statistic, but this does not make a zero-reference-variance standardized endpoint count well defined.
