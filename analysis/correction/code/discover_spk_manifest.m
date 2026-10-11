function manifest = discover_spk_manifest(dataDir, excludedRecording)
%DISCOVER_SPK_MANIFEST Build the 6-well longitudinal recording manifest.

if nargin < 2 || strlength(string(excludedRecording)) == 0
    excludedRecording = "";
end

listing = dir(fullfile(dataDir, '**', '*.spk'));
if isempty(listing)
    error('No .spk files found under %s', dataDir);
end

spk_path = strings(0, 1);
source_file = strings(0, 1);
DIV = zeros(0, 1);
plate_index = zeros(0, 1);
well_row = zeros(0, 1);
well_col = zeros(0, 1);
well_label = strings(0, 1);
recording = strings(0, 1);

for fileIdx = 1:numel(listing)
    pathValue = string(fullfile(listing(fileIdx).folder, listing(fileIdx).name));
    [~, base, ~] = fileparts(listing(fileIdx).name);
    tokens = regexp(base, '^WT[-_]DIV(?<div>\d+)_(?<plate>[123])$', 'names');
    if isempty(tokens)
        error('Could not parse DIV and plate from %s', listing(fileIdx).name);
    end
    divValue = str2double(tokens.div);
    plateValue = str2double(tokens.plate);
    for row = 1:2
        for col = 1:3
            label = sprintf('%c%d', char('A' + row - 1), col);
            spk_path(end+1, 1) = pathValue; %#ok<AGROW>
            source_file(end+1, 1) = string(listing(fileIdx).name); %#ok<AGROW>
            DIV(end+1, 1) = divValue; %#ok<AGROW>
            plate_index(end+1, 1) = plateValue; %#ok<AGROW>
            well_row(end+1, 1) = row; %#ok<AGROW>
            well_col(end+1, 1) = col; %#ok<AGROW>
            well_label(end+1, 1) = string(label); %#ok<AGROW>
            recording(end+1, 1) = string(sprintf('WT_DIV%02d_P%d_%s', ...
                divValue, plateValue, label)); %#ok<AGROW>
        end
    end
end

excluded = recording == string(excludedRecording);
manifest = table(recording, DIV, plate_index, well_row, well_col, well_label, ...
    source_file, spk_path, excluded);
manifest = sortrows(manifest, {'DIV', 'plate_index', 'well_row', 'well_col'});
end

