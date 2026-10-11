function fit_synthetic_maxent(inputPath, outputPath, maxentDir, fitSeed, sampleSeed)
% Original fit, with explicit independent native sampling seed for this audit.
maxNumCompThreads(2);
rng(double(fitSeed), 'twister');
input = load(inputPath);
x = input.x;
options = struct('threshold', 2, 'max_steps', 2000, 'burnin', 10000, ...
    'silent', true, 'fit_kpairwise', false);
lastwarn('');
fit = fit_maxent_recording(x, maxentDir, 20, size(x, 2), options);
[fit_warning, fit_warning_id] = lastwarn;
% N16 fitting is exhaustive/deterministic. Original unseeded pool is discarded.
fit.pairwise_samples = uint8(maxent.generateSamples(fit.pairwise_model, ...
    20 * size(x, 2), 'burnin', 10000, 'separation', size(x, 1), ...
    'randseed', sampleSeed));
d = double(x); s = double(fit.pairwise_samples);
fit.pairwise_firing_rmse = sqrt(mean((mean(d, 2) - mean(s, 2)).^2));
dp = d*d'/size(d, 2); sp = s*s'/size(s, 2);
upper = triu(true(size(dp)), 1);
fit.pairwise_pair_rmse = sqrt(mean((dp(upper)-sp(upper)).^2));
edges = -0.5:1:(size(x, 1)+0.5);
dk = histcounts(sum(d, 1), edges, 'Normalization', 'probability');
sk = histcounts(sum(s, 1), edges, 'Normalization', 'probability');
fit.pairwise_k_rmse = sqrt(mean((dk-sk).^2));
% Audit native seed mapping without advancing any retained sampling stream.
a = maxent.generateSamples(fit.pairwise_model, 128, 'randseed', sampleSeed);
b = maxent.generateSamples(fit.pairwise_model, 128, 'randseed', sampleSeed);
c = maxent.generateSamples(fit.pairwise_model, 128, 'randseed', mod(sampleSeed+1, 2^32));
native_seed_repeat_equal = isequal(a,b);
native_seed_change_different = ~isequal(a,c);
assert(native_seed_repeat_equal && native_seed_change_different, 'Native seed audit failed');
save(outputPath, 'fit', 'fitSeed', 'sampleSeed', 'fit_warning', 'fit_warning_id', ...
    'native_seed_repeat_equal', 'native_seed_change_different', '-v7');
end
