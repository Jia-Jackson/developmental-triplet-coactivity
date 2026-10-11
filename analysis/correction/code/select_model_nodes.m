function selected = select_model_nodes(x, minStateBins, maxNodes)
%SELECT_MODEL_NODES Select the most active nodes with both binary states observed.

if nargin < 2 || isempty(minStateBins), minStateBins = 20; end
if nargin < 3 || isempty(maxNodes), maxNodes = inf; end
counts = sum(logical(x), 2);
eligible = find(counts >= minStateBins & counts <= size(x, 2) - minStateBins);
if isfinite(maxNodes) && numel(eligible) > maxNodes
    ranking = sortrows([eligible, counts(eligible)], [-2 1]);
    selected = sort(ranking(1:maxNodes, 1));
else
    selected = eligible;
end
selected = selected(:);
end
