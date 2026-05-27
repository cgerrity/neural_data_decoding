% scripts/convert_reference_cm_tables.m
% One-shot fixture-generation helper.
%
% Reads the MATLAB-native `CM_Table.mat` / `CM_Table_Validation.mat` files
% that live under `tests/fixtures/reference_cm_tables/`, calls
% `table2struct(T, 'ToScalar', true)` on each (producing a single struct
% whose fields are column arrays), and saves the result as a v7 (non-HDF5)
% `.mat` file that scipy.io.loadmat can read.
%
% The struct is saved under the variable name `CM_Table`, matching the
% Python writer's convention so a single parity test can compare both.
%
% Run from MATLAB:
%     >> run('scripts/convert_reference_cm_tables.m')
%
% Or from a shell:
%     $ matlab -batch "run('scripts/convert_reference_cm_tables.m')"
%
% This is a one-shot operation — re-run only when the underlying MATLAB
% fixtures change. The outputs are gitignored along with the rest of
% tests/fixtures/reference_cm_tables/*.mat.

thisDir   = fileparts(mfilename('fullpath'));
fixtureDir = fullfile(thisDir, '..', 'tests', 'fixtures', 'reference_cm_tables');

inputs = {'CM_Table.mat', 'CM_Table_Validation.mat'};

for k = 1:numel(inputs)
    src   = inputs{k};
    inPath  = fullfile(fixtureDir, src);
    if ~isfile(inPath)
        warning('convert_reference_cm_tables:missingInput', ...
            'Skipping %s — file not found.', inPath);
        continue;
    end
    [~, base, ~] = fileparts(src);
    outPath = fullfile(fixtureDir, [base '_python_struct.mat']);

    payload = load(inPath);
    if ~isfield(payload, 'CM_Table')
        warning('convert_reference_cm_tables:missingVariable', ...
            'Skipping %s — no variable named CM_Table.', inPath);
        continue;
    end
    T = payload.CM_Table;
    CM_Table = table2struct(T, 'ToScalar', true); %#ok<NASGU>
    save(outPath, 'CM_Table', '-v7');
    fprintf('Wrote %s (%d trials, %d fields)\n', outPath, ...
        height(T), numel(fieldnames(CM_Table)));
end
