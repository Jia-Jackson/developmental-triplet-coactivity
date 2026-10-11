function tests = test_conversion_manifest
tests = functiontests(localfunctions);
end

function testDiscoversAllBoardsAndAppliesOneExclusion(testCase)
dataDir = 'historical_local_path/DATA';
excluded = "WT_DIV35_P2_A1";
manifest = discover_spk_manifest(dataDir, excluded);

verifyEqual(testCase, height(unique(manifest(:, {'spk_path'}))), 24);
verifyEqual(testCase, height(manifest), 144);
verifyEqual(testCase, sum(manifest.excluded), 1);
verifyEqual(testCase, string(manifest.recording(manifest.excluded)), excluded);
verifyEqual(testCase, sum(~manifest.excluded), 143);
verifyEqual(testCase, sort(unique(manifest.DIV))', [14 21 28 35 42 49 56 63]);
verifyEqual(testCase, sort(unique(manifest.plate_index))', [1 2 3]);
end

