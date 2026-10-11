function valid = requested_record_eligible(manifest, recording, analysis, requestedN)
% Explicit requested-N decision; absent/duplicate entries are errors, not exclusions.
required = {'recording','analysis','requested_n','eligible'};
if ~istable(manifest) || ~all(ismember(required, manifest.Properties.VariableNames))
    error('Invalid requested-N eligibility table');
end
rows = manifest(string(manifest.recording) == string(recording) & ...
    string(manifest.analysis) == string(analysis), :);
if height(rows) ~= 1 || double(rows.requested_n) ~= requestedN
    error('Missing, duplicate or mismatched eligibility for %s / %s', recording, analysis);
end
value = lower(string(rows.eligible));
if ~ismember(value, ["true", "false", "1", "0"])
    error('Invalid eligibility boolean');
end
valid = ismember(value, ["true", "1"]);
end
