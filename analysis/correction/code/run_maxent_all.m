function run_maxent_all(resultDir, maxentDir, maxFiles, maxModelNodes, outputFolder, workerIndex, workerCount, nodeManifestPath)
%RUN_MAXENT_ALL Fit pairwise and K-pairwise nulls to all 25-ms recordings.

if nargin < 3 || isempty(maxFiles), maxFiles = inf; end
if nargin < 4 || isempty(maxModelNodes), maxModelNodes = inf; end
if nargin < 5 || isempty(outputFolder), outputFolder = 'maxent_models'; end
if nargin < 6 || isempty(workerIndex), workerIndex = 1; end
if nargin < 7 || isempty(workerCount), workerCount = 1; end
if nargin < 8, nodeManifestPath = ''; end
eligibilityPath = fullfile(resultDir, 'tables', 'computational_closure_v2', 'requested_node_eligibility.csv');
eligibility = readtable(eligibilityPath, 'TextType', 'string');
if isempty(nodeManifestPath)
    analysis = sprintf('dynamic%d', maxModelNodes);
else
    analysis = sprintf('fixed%d', maxModelNodes);
end
if ~isempty(nodeManifestPath)
    if exist(nodeManifestPath, 'file') ~= 2
        error('Node manifest not found: %s', nodeManifestPath);
    end
    nodeManifest = readtable(nodeManifestPath, 'TextType', 'string');
else
    nodeManifest = table();
end
binaryDir = fullfile(resultDir, 'data_intermediate', 'binary_25ms');
outputDir = fullfile(resultDir, 'data_intermediate', outputFolder);
if exist(outputDir, 'dir') ~= 7, mkdir(outputDir); end
files = dir(fullfile(binaryDir, '*_25ms.mat'));
[~, order] = sort({files.name});
files = files(order);
files = files(1:min(numel(files), maxFiles));
allFileCount = numel(files);
files = files(partition_indices(allFileCount, workerIndex, workerCount));
if isempty(files), error('No 25-ms binary MAT files found.'); end

options = struct('threshold', 2.0, 'max_steps', 2000, ...
    'kpairwise_max_steps', 5000, 'max_nsamples', 5000, ...
    'burnin', 10000, 'silent', true, 'fit_kpairwise', false);
nBlocks = 20;

for idx = 1:numel(files)
    inputPath = fullfile(files(idx).folder, files(idx).name);
    [~, base, ~] = fileparts(files(idx).name);
    recording = erase(base, '_25ms');
    outputPath = fullfile(outputDir, [recording '_maxent.mat']);
    if ~requested_record_eligible(eligibility, recording, analysis, maxModelNodes)
        if exist(outputPath, 'file') == 2
            error('Unexpected cached model for ineligible %s / %s', recording, analysis);
        end
        fprintf('[worker %d] Unavailable %s / %s: requested-node eligibility\n', workerIndex, recording, analysis);
        continue
    end
    if exist(outputPath, 'file') == 2
    fprintf('[worker %d, %03d/%03d] Existing %s\n', workerIndex, idx, numel(files), recording);
        continue
    end
    fprintf('[worker %d, %03d/%03d] Fitting %s\n', workerIndex, idx, numel(files), recording);
    input = load(inputPath, 'x', 'DIV', 'plate', 'well');
    xFull = logical(input.x);
    if isempty(nodeManifestPath)
        active_nodes = select_model_nodes(xFull, 20, maxModelNodes);
        selection_rule = 'highest active-bin counts among nodes with >=20 active and >=20 inactive bins; ties by electrode index';
    else
        active_nodes = select_manifest_nodes(nodeManifest, recording, maxModelNodes);
        selection_rule = sprintf('fixed longitudinal maximin node manifest; N=%d', maxModelNodes);
    end
    if numel(active_nodes) ~= maxModelNodes
        error('Requested-node contract mismatch for %s: found %d, required %d', recording, numel(active_nodes), maxModelNodes);
    end
    x = xFull(active_nodes, :);
    globalIndex = (idx - 1) * workerCount + workerIndex;
    rng(100000 + globalIndex, 'twister');
    fit = fit_maxent_recording(x, maxentDir, nBlocks, size(x, 2), options);
    skipped = false;
    max_model_nodes = maxModelNodes;
    toolbox_commit = '82b4528675fed43beb1d90a1c06b43189afe023e';
    node_manifest_path = nodeManifestPath;
    save(outputPath, 'recording', 'active_nodes', 'fit', 'skipped', ...
        'max_model_nodes', 'selection_rule', 'toolbox_commit', 'node_manifest_path', '-v7');
end
end
