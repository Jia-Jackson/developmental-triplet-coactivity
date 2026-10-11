function indices = partition_indices(nItems, workerIndex, workerCount)
%PARTITION_INDICES Return deterministic one-based strided worker indices.

if workerCount < 1 || workerIndex < 1 || workerIndex > workerCount
    error('Invalid worker partition.');
end
indices = find(mod((1:nItems) - 1, workerCount) == workerIndex - 1);
end
