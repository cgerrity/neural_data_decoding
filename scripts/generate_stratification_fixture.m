function generate_stratification_fixture()
%GENERATE_STRATIFICATION_FIXTURE  Build a stratification reference fixture.
%   Creates a synthetic identifier table, runs MATLAB's recursive
%   stratification on it, and saves both the input + output to
%   tests/fixtures/reference_partitions/synthetic_easy_partition.mat so
%   the Python parity test can compare against a known-good MATLAB output.
%
%   The fixture is self-contained: it does NOT require any preprocessed
%   neural data, only the MATLAB pipeline source (Processing_Functions_cgg).
%   Regenerate any time by running this script in MATLAB or via:
%
%       python scripts/prepare_golden_fixtures.py --milestone 0
%
%   The expected output of the Python `stratify()` function (with the same
%   IdentifierTable + AllSplitNames + NumFolds) must equal the saved
%   PartitionGroups vector trial-for-trial.

%% Resolve paths via the shared helper (respects NDD_MATLAB_SOURCE_ROOT).
script_dir = fileparts(mfilename('fullpath'));
repo_root  = fileparts(script_dir);
addpath(script_dir);
ndd_add_matlab_paths();

%% Reproducible synthetic identifier table
% A small, hand-shaped dataset that exercises both the "maintain" branch
% (small categories collapsed into a single leaf) and the recursive
% "further split" branch.  Numeric encodings throughout because MATLAB's
% Identifiers cell array requires each cell to be a row of scalars.
rng(42);

num_trials = 60;
data_numbers = (1:num_trials)';

% Level 1 columns: two binary "dimension" flags.
dim1 = double(randi([0, 1], num_trials, 1));
dim2 = double(randi([0, 1], num_trials, 1));

% Level 2 column: binary correctness flag.
correct = double(randi([0, 1], num_trials, 1));

% Level 3 column: integer session id (3 sessions).
session = double(randi([1, 3], num_trials, 1));

identifier_columns = ["Data Number", "Dimension 1", "Dimension 2", ...
                     "Correct Trial", "Session Name"];
identifier_data    = [data_numbers, dim1, dim2, correct, session];

% Build the cell-array form MATLAB's stratification helpers expect.
Identifiers = cell(num_trials, 1);
for trial = 1:num_trials
    Identifiers{trial} = identifier_data(trial, :);
end
IdentifierName = identifier_columns;

%% Stratification hierarchy
% Three-level hierarchy chosen to exercise both code paths in
% cgg_procSplitIntoGroups: level 1 produces some leaves AND some recurse,
% level 2 produces both, level 3 forces the remaining recursions to bottom
% out at the deepest level.
AllSplitNames = {
    ["Dimension 1", "Dimension 2"];   % level 1: 4 cross-product cells
    "Correct Trial";                  % level 2: 2 cells
    "Session Name"                    % level 3: up to 3 cells
};

NumFolds = 5;

%% Run the MATLAB stratifier
%
% cgg_procAssignGroupsBySplit returns a cell array of leaf groups (one
% entry per leaf, each entry is the list of DataNumbers in that leaf).
% We replicate the post-processing from cgg_procAssignGroups (the wrapper)
% to map those leaf groups onto a per-trial PartitionGroups vector.

GroupList = cgg_procAssignGroupsBySplit(Identifiers, IdentifierName, ...
    AllSplitNames, NumFolds);

PartitionGroups = nan(1, num_trials);
data_number_lookup = data_numbers;
for group_idx = 1:numel(GroupList)
    member_data_numbers = GroupList{group_idx};
    PartitionGroups(ismember(data_number_lookup, member_data_numbers)) = group_idx;
end

% Sanity check: every trial must have been assigned to a group.
if any(isnan(PartitionGroups))
    unassigned = find(isnan(PartitionGroups));
    error('generate_stratification_fixture:unassigned_trial', ...
        '%d trial(s) left unassigned by MATLAB stratifier: %s', ...
        numel(unassigned), mat2str(unassigned));
end

PartitionGroups = PartitionGroups(:);     % column for cleaner Python load

%% Pack the fixture
% Use simple types (matrices + cell arrays of char) so scipy.io.loadmat
% can round-trip cleanly without needing mat73.
fixture = struct();
fixture.IdentifierColumns       = cellstr(identifier_columns);
fixture.IdentifierData          = identifier_data;
fixture.AllSplitNamesLevel1     = cellstr(AllSplitNames{1});
fixture.AllSplitNamesLevel2     = cellstr(AllSplitNames{2});
fixture.AllSplitNamesLevel3     = cellstr(AllSplitNames{3});
fixture.NumSplitLevels          = numel(AllSplitNames);
fixture.NumFolds                = NumFolds;
fixture.PartitionGroups         = PartitionGroups;
fixture.SchemaVersion           = 1;
fixture.GeneratorScript         = mfilename('fullpath');

%% Save
output_dir = fullfile(repo_root, 'tests', 'fixtures', 'reference_partitions');
if ~isfolder(output_dir)
    mkdir(output_dir);
end
output_path = fullfile(output_dir, 'synthetic_easy_partition.mat');

% Default save format is whatever MATLAB's current setting is; force v7
% (not v7.3) so scipy.io.loadmat can read it without mat73.
save(output_path, '-struct', 'fixture', '-v7');

fprintf('Saved stratification fixture:\n  %s\n', output_path);
fprintf('  num_trials        = %d\n', num_trials);
fprintf('  num_split_levels  = %d\n', numel(AllSplitNames));
fprintf('  num_folds         = %d\n', NumFolds);
fprintf('  num_unique_groups = %d\n', numel(unique(PartitionGroups)));

end
