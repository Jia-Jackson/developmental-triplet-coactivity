function convert_spk_to_intermediate(dataDir, resultDir, axionDir, excludedRecording)
%CONVERT_SPK_TO_INTERMEDIATE Read Axion SPK events and save one MAT per well.
% This conversion does not redetect spikes or modify AxIS event times.

if nargin < 4 || strlength(string(excludedRecording)) == 0
    excludedRecording = "WT_DIV35_P2_A1";
end
if exist(axionDir, 'dir') ~= 7
    error('AxionFileLoader directory not found: %s', axionDir);
end
addpath(axionDir);

outArrayDir = fullfile(resultDir, 'data_intermediate', 'spike_arrays');
outMetaDir = fullfile(resultDir, 'data_intermediate', 'metadata');
if exist(outArrayDir, 'dir') ~= 7, mkdir(outArrayDir); end
if exist(outMetaDir, 'dir') ~= 7, mkdir(outMetaDir); end

manifest = discover_spk_manifest(dataDir, excludedRecording);
writetable(manifest, fullfile(outMetaDir, 'recording_manifest_all.csv'));

[channel_id, electrode_row, electrode_col] = axion64_order();
retained = manifest(~manifest.excluded, :);
nRows = height(retained);
n_spikes = zeros(nRows, 1);
n_active_electrodes_0p01Hz = zeros(nRows, 1);
mean_firing_rate_hz_all = zeros(nRows, 1);
duration_s_meta = zeros(nRows, 1);
fs_meta = zeros(nRows, 1);
output_arrays_mat = strings(nRows, 1);

uniquePaths = unique(retained.spk_path, 'stable');
for fileIdx = 1:numel(uniquePaths)
    spkPath = uniquePaths(fileIdx);
    fprintf('[%02d/%02d] Loading %s\n', fileIdx, numel(uniquePaths), spkPath);
    axisFile = AxisFile(char(spkPath));
    spikeSet = axisFile.SpikeData;
    duration_s = double(spikeSet.Duration);
    fs = double(spikeSet.SamplingFrequency);
    spikeData = spikeSet.LoadData;
    dims = size(spikeData);
    if numel(dims) < 4 || dims(1) ~= 2 || dims(2) ~= 3
        error('Expected a 2 x 3 well SPK array in %s', spkPath);
    end

    rowsForFile = find(retained.spk_path == spkPath);
    for localIdx = 1:numel(rowsForFile)
        metaIdx = rowsForFile(localIdx);
        wellRow = retained.well_row(metaIdx);
        wellCol = retained.well_col(metaIdx);
        recording = retained.recording(metaIdx);

        channel_counts = zeros(64, 1);
        channel_firing_rate_hz = zeros(64, 1);
        spike_time_s = zeros(0, 1);
        spike_channel_index = zeros(0, 1);
        spike_channel_id = zeros(0, 1);
        spike_electrode_row = zeros(0, 1);
        spike_electrode_col = zeros(0, 1);

        for channelIdx = 1:64
            eRow = electrode_row(channelIdx);
            eCol = electrode_col(channelIdx);
            channelSpikes = spikeData{wellRow, wellCol, eCol, eRow};
            count = numel(channelSpikes);
            channel_counts(channelIdx) = count;
            channel_firing_rate_hz(channelIdx) = count / duration_s;
            if count == 0, continue; end
            times = zeros(count, 1);
            for spikeIdx = 1:count
                times(spikeIdx) = double(channelSpikes(spikeIdx).Start) + ...
                    double(channelSpikes(spikeIdx).TriggerSampleOffset) / fs;
            end
            spike_time_s = [spike_time_s; times]; %#ok<AGROW>
            spike_channel_index = [spike_channel_index; repmat(channelIdx, count, 1)]; %#ok<AGROW>
            spike_channel_id = [spike_channel_id; repmat(channel_id(channelIdx), count, 1)]; %#ok<AGROW>
            spike_electrode_row = [spike_electrode_row; repmat(eRow, count, 1)]; %#ok<AGROW>
            spike_electrode_col = [spike_electrode_col; repmat(eCol, count, 1)]; %#ok<AGROW>
        end

        [spike_time_s, order] = sort(spike_time_s);
        spike_channel_index = spike_channel_index(order);
        spike_channel_id = spike_channel_id(order);
        spike_electrode_row = spike_electrode_row(order);
        spike_electrode_col = spike_electrode_col(order);

        channel_index = (1:64)';
        divValue = retained.DIV(metaIdx);
        plateIndex = retained.plate_index(metaIdx);
        wellLabel = retained.well_label(metaIdx);
        sourceSpkPath = spkPath;
        outputPath = fullfile(outArrayDir, char(recording + "_arrays.mat"));
        save(outputPath, 'spike_time_s', 'spike_channel_index', 'spike_channel_id', ...
            'spike_electrode_row', 'spike_electrode_col', 'channel_index', ...
            'channel_id', 'electrode_row', 'electrode_col', 'channel_counts', ...
            'channel_firing_rate_hz', 'duration_s', 'fs', 'recording', ...
            'divValue', 'plateIndex', 'wellRow', 'wellCol', 'wellLabel', ...
            'sourceSpkPath', '-v7');

        n_spikes(metaIdx) = numel(spike_time_s);
        n_active_electrodes_0p01Hz(metaIdx) = sum(channel_firing_rate_hz >= 0.01);
        mean_firing_rate_hz_all(metaIdx) = mean(channel_firing_rate_hz);
        duration_s_meta(metaIdx) = duration_s;
        fs_meta(metaIdx) = fs;
        output_arrays_mat(metaIdx) = string(outputPath);
    end
    clear spikeData spikeSet axisFile
end

recording_metadata = addvars(retained, n_spikes, n_active_electrodes_0p01Hz, ...
    mean_firing_rate_hz_all, duration_s_meta, fs_meta, output_arrays_mat);
writetable(recording_metadata, fullfile(outMetaDir, 'recording_metadata.csv'));
fprintf('Converted %d retained well recordings.\n', height(recording_metadata));
end

function [channels, rows, cols] = axion64_order()
channels = zeros(64, 1);
rows = zeros(64, 1);
cols = zeros(64, 1);
idx = 1;
for row = 1:8
    for col = 1:8
        channels(idx) = col * 10 + row;
        rows(idx) = row;
        cols(idx) = col;
        idx = idx + 1;
    end
end
end

