function selected = select_manifest_nodes(manifest, recording, nNodes)
%SELECT_MANIFEST_NODES Read one deterministic fixed-node set from a manifest table.

if ~istable(manifest)
    error('Manifest must be a MATLAB table.');
end
required = {'recording', 'n_nodes', 'fixed_rank', 'electrode_index_one_based'};
if ~all(ismember(required, manifest.Properties.VariableNames))
    error('Manifest is missing required columns.');
end
mask = string(manifest.recording) == string(recording) & ...
    double(manifest.n_nodes) == double(nNodes);
rows = manifest(mask, :);
if height(rows) ~= nNodes
    error('Expected %d manifest nodes for %s, found %d.', nNodes, recording, height(rows));
end
rows = sortrows(rows, 'fixed_rank', 'ascend');
selected = double(rows.electrode_index_one_based(:));
if numel(unique(selected)) ~= nNodes || any(selected < 1) || any(selected > 64)
    error('Manifest nodes for %s are invalid or duplicated.', recording);
end
end
