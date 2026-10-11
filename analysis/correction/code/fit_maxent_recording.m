function result = fit_maxent_recording(x, maxentDir, nBlocks, blockLength, options)
%FIT_MAXENT_RECORDING Fit pairwise and K-pairwise models and draw null data.

if nargin < 5, options = struct(); end
options = defaults(options);
if exist(maxentDir, 'dir') ~= 7
    error('Maximum-entropy toolbox not found: %s', maxentDir);
end
addpath(maxentDir);
x = logical(x);
[nNodes, nBins] = size(x);
if nNodes < 2 || nBins < 10
    error('Insufficient binary data for maximum-entropy fitting.');
end

pairwiseModel = maxent.createModel(nNodes, 'pairwise');
[pairwiseModel, pairwiseConverged] = trainOne(pairwiseModel, x, options);
pairwiseSamples = maxent.generateSamples(pairwiseModel, nBlocks * blockLength, ...
    'burnin', options.burnin, 'separation', nNodes);

if options.fit_kpairwise
    kpairwiseModel = maxent.createModel(nNodes, 'kpairwise');
    % The toolbox stores N+1 synchrony factors first, followed by all Ising
    % factors. Warm-start only the Ising block from the fitted pairwise model.
    nPairFactors = numel(pairwiseModel.factors);
    kpairwiseModel.factors((nNodes + 2):(nNodes + 1 + nPairFactors)) = ...
        pairwiseModel.factors;
    kOptions = options;
    kOptions.max_steps = options.kpairwise_max_steps;
    [kpairwiseModel, kpairwiseConverged] = trainOne(kpairwiseModel, x, kOptions);
    kpairwiseSamples = maxent.generateSamples(kpairwiseModel, nBlocks * blockLength, ...
        'burnin', options.burnin, 'separation', nNodes);
else
    kpairwiseModel = struct();
    kpairwiseConverged = false;
    kpairwiseSamples = zeros(nNodes, 0, 'uint8');
end

[pairFiringRmse, pairPairRmse, pairKRmse] = diagnosticRmse(x, pairwiseSamples);
if options.fit_kpairwise
    [kpairFiringRmse, kpairPairRmse, kpairKRmse] = diagnosticRmse(x, kpairwiseSamples);
else
    kpairFiringRmse = nan;
    kpairPairRmse = nan;
    kpairKRmse = nan;
end

result = struct();
result.n_nodes = nNodes;
result.n_data_bins = nBins;
result.n_blocks = nBlocks;
result.block_length = blockLength;
result.pairwise_model = pairwiseModel;
result.kpairwise_model = kpairwiseModel;
result.pairwise_converged = logical(pairwiseConverged);
result.kpairwise_converged = logical(kpairwiseConverged);
result.pairwise_samples = uint8(pairwiseSamples);
result.kpairwise_samples = uint8(kpairwiseSamples);
result.pairwise_firing_rmse = pairFiringRmse;
result.pairwise_pair_rmse = pairPairRmse;
result.pairwise_k_rmse = pairKRmse;
result.kpairwise_firing_rmse = kpairFiringRmse;
result.kpairwise_pair_rmse = kpairPairRmse;
result.kpairwise_k_rmse = kpairKRmse;
result.options = options;
end

function [model, converged] = trainOne(model, x, options)
if size(x, 1) <= 25
    [model, converged] = maxent.trainModel(model, x, ...
        'threshold', options.threshold, 'max_steps', options.max_steps, ...
        'silent', options.silent);
else
    [model, converged] = maxent.trainModel(model, x, ...
        'threshold', options.threshold, 'max_steps', options.max_steps, ...
        'max_nsamples', options.max_nsamples, 'silent', options.silent);
end
end

function options = defaults(options)
values = struct('threshold', 2.0, 'max_steps', 2000, ...
    'kpairwise_max_steps', 5000, 'max_nsamples', 5000, ...
    'burnin', 10000, 'silent', true, 'fit_kpairwise', true);
names = fieldnames(values);
for idx = 1:numel(names)
    if ~isfield(options, names{idx})
        options.(names{idx}) = values.(names{idx});
    end
end
end

function [firingRmse, pairRmse, kRmse] = diagnosticRmse(data, simulated)
data = double(data);
simulated = double(simulated);
firingRmse = sqrt(mean((mean(data, 2) - mean(simulated, 2)).^2));
dataPairs = data * data' / size(data, 2);
simPairs = simulated * simulated' / size(simulated, 2);
upper = triu(true(size(dataPairs)), 1);
pairRmse = sqrt(mean((dataPairs(upper) - simPairs(upper)).^2));
nNodes = size(data, 1);
dataK = histcounts(sum(data, 1), -0.5:1:(nNodes + 0.5), 'Normalization', 'probability');
simK = histcounts(sum(simulated, 1), -0.5:1:(nNodes + 0.5), 'Normalization', 'probability');
kRmse = sqrt(mean((dataK - simK).^2));
end
