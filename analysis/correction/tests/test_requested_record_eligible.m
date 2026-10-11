function test_requested_record_eligible()
t = table(["r"; "r"; "s"], ["dynamic16"; "fixed12"; "dynamic16"], ...
    [16;12;16], [false;true;true], ...
    'VariableNames', {'recording','analysis','requested_n','eligible'});
assert(~requested_record_eligible(t, 'r', 'dynamic16', 16));
assert(requested_record_eligible(t, 'r', 'fixed12', 12));
assert(requested_record_eligible(t, 's', 'dynamic16', 16));
failed = false;
try
    requested_record_eligible(t, 'absent', 'dynamic16', 16);
catch
    failed = true;
end
assert(failed, 'Missing eligibility rows must raise');
disp('REQUESTED_RECORD_ELIGIBLE_TEST_PASS');
end
