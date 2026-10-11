function benchmark_fixed_n20(resultDir, maxentDir, manifestPath, recordings)
%BENCHMARK_FIXED_N20 Benchmark exact N=20 fitting on sparse/median/dense records.

if nargin < 4 || isempty(recordings)
    recordings = {'WT_DIV14_P3_A1', 'WT_DIV21_P1_B1', 'WT_DIV42_P1_A1'};
end
manifest = readtable(manifestPath, 'TextType', 'string');
outputDir = fullfile(resultDir, 'data_intermediate', 'maxent_fixed_n20_benchmark');
if exist(outputDir, 'dir') ~= 7, mkdir(outputDir); end
options = struct('threshold', 2.0, 'max_steps', 2000, ...
    'kpairwise_max_steps', 5000, 'max_nsamples', 5000, ...
    'burnin', 10000, 'silent', true, 'fit_kpairwise', false);
nBlocks = 20;
rows = cell(numel(recordings), 1);
for idx = 1:numel(recordings)
    recording = recordings{idx};
    inputPath = fullfile(resultDir, 'data_intermediate', 'binary_25ms', [recording '_25ms.mat']);
    input = load(inputPath, 'x');
    nodes = select_manifest_nodes(manifest, recording, 20);
    x = logical(input.x(nodes, :));
    rng(20260828 + idx, 'twister');
    before = memory;
    started = tic;
    fit = fit_maxent_recording(x, maxentDir, nBlocks, size(x, 2), options);
    elapsedSeconds = toc(started);
    after = memory;
    observedWorkingSet = max(before.MemUsedMATLAB, after.MemUsedMATLAB);
    outputPath = fullfile(outputDir, [recording '_maxent.mat']);
    save(outputPath, 'recording', 'nodes', 'fit', 'elapsedSeconds', ...
        'observedWorkingSet', '-v7');
    rows{idx} = table(string(recording), elapsedSeconds, observedWorkingSet, ...
        logical(fit.pairwise_converged), fit.pairwise_firing_rmse, ...
        fit.pairwise_pair_rmse, fit.pairwise_k_rmse, ...
        'VariableNames', {'recording','elapsed_seconds','observed_working_set_bytes', ...
        'converged','firing_rmse','pairwise_rmse','k_rmse'});
    fprintf('[%d/%d] %s completed in %.1f s\n', idx, numel(recordings), recording, elapsedSeconds);
end
summary = vertcat(rows{:});
writetable(summary, fullfile(resultDir, 'tables', 'computational_closure_v2', ...
    'v2_fixed_n20_exact_benchmark.csv'));
end
