function tests = test_maxent_integration
tests = functiontests(localfunctions);
end

function testFitsAndSamplesPairwiseAndKPairwiseModels(testCase)
rng(7);
x = rand(4, 600) < [0.20; 0.35; 0.50; 0.65];
maxentDir = 'historical_local_path/maxent_toolbox';
options = struct('threshold', 1.5, 'max_steps', 200, ...
    'max_nsamples', 1000, 'burnin', 1000, 'silent', true);
result = fit_maxent_recording(x, maxentDir, 2, 200, options);

verifyEqual(testCase, size(result.pairwise_samples), [4 400]);
verifyEqual(testCase, size(result.kpairwise_samples), [4 400]);
verifyTrue(testCase, isfinite(result.pairwise_firing_rmse));
verifyTrue(testCase, isfinite(result.pairwise_pair_rmse));
verifyTrue(testCase, isfinite(result.kpairwise_k_rmse));
end

function testSelectsHighestCountEligibleNodesDeterministically(testCase)
x = false(5, 100);
x(1, 1:30) = true;
x(2, 1:50) = true;
x(3, 1:10) = true;
x(4, 1:40) = true;
x(5, 1:50) = true;
selected = select_model_nodes(x, 20, 2);

verifyEqual(testCase, selected(:)', [2 5]);
end

function testPartitionIndicesCoverInputsWithoutOverlap(testCase)
parts = cell(1, 4);
for worker = 1:4
    parts{worker} = partition_indices(11, worker, 4);
end
combined = sort([parts{:}]);

verifyEqual(testCase, combined, 1:11);
verifyEqual(testCase, numel(unique(combined)), 11);
end

function testManifestSelectionUsesRequestedRecordingAndNodeCount(testCase)
manifest = table(...
    ["WT_DIV14_P1_A1"; "WT_DIV14_P1_A1"; "WT_DIV14_P1_A1"; "WT_DIV14_P1_A1"], ...
    [2; 2; 3; 3], [2; 1; 2; 1], [5; 3; 8; 4], ...
    'VariableNames', {'recording', 'n_nodes', 'fixed_rank', 'electrode_index_one_based'});
selected = select_manifest_nodes(manifest, 'WT_DIV14_P1_A1', 2);
verifyEqual(testCase, selected(:)', [3 5]);
end
