% scripts/generate_t2_confidence_beta_fixture.m
%
% Milestone C #7 — Confidence_Beta P-controller parity fixture.
%
% Iterates cgg_getConfidenceLossInformation over a known sequence of
% (TrialConfidence, TaskConfidence) batches and captures the updated
% Confidence_Beta + the three EMA values after each call. The Python
% parity test reproduces every value to ~1e-12.
%
% The Beta update is documented in cgg_getConfidenceLossInformation.m
% lines 60-75 (despite the function name, it's a pure P-controller):
%   diff       = ConfidenceTarget - batchMeanTotal     (target = 0.5)
%   beta_next  = beta_prev * (1 + diff * rate)         (rate = 1.0)
%   beta_next  = clamp(beta_next, [0.1, 10])
%
% The EMAs use BatchFraction (γ) as the smoothing factor:
%   ema_next = (1 - γ) * ema_prev + γ * batch_mean
%
% Three batches with distinct confidence distributions cover both
% directions of Beta (high mean → Beta down; low mean → Beta up) and
% confirm the clamp doesn't fire on these (gentle) inputs.

clear; close all;

% ───────────────────────── Path setup ─────────────────────────
addpath(fileparts(mfilename('fullpath')));
ndd_add_matlab_paths();
thisDir = fileparts(mfilename('fullpath'));
projDir = fullfile(thisDir, '..');

rng(77, 'twister');  % seed for reproducibility

% ───────────────────────── Shared inputs ─────────────────────────
K = 3;   % output dimensions (matches the cgg_getConfidenceLossInformation
         % ValidClassificationIndices convention)
B = 4;   % batch
T = 5;   % time
BatchFraction = 0.25;
ValidClassificationIndices = true(1, K);

% Each batch generates fresh confidence arrays with controlled means.
batch_means_total = [0.8, 0.3, 0.6];   % drives Beta down, up, then down
NumBatches = numel(batch_means_total);

% Pre-allocate per-batch arrays so the Python test can replay them.
batches = cell(NumBatches, 1);
for bidx = 1:NumBatches
    target_mean = batch_means_total(bidx);
    % Generate trial in [target-0.15, target+0.15] (clipped to [0.05, 0.95]).
    trial_raw = max(min(target_mean + 0.3*(rand(1, B, T) - 0.5), 0.95), 0.05);
    % Task = target / mean(trial) to land TotalConfidence's mean at target.
    task_raw  = max(min(target_mean ./ trial_raw, 0.95), 0.05);
    % Tile task across K dims (typical per-dim confidence shape).
    task_arr = repmat(task_raw, K, 1, 1);

    trial_dl = dlarray(trial_raw, 'CBT');
    task_dl  = dlarray(task_arr,  'CBT');

    batches{bidx} = struct( ...
        'trial_in', trial_raw, ...
        'task_in',  task_arr, ...
        'trial_dl', trial_dl, ...
        'task_dl',  task_dl);
end

% ───────────────────────── Iterate the controller ─────────────────────────
LossInformation = struct();
LossInformation.DatasetTotalConfidence = NaN;
LossInformation.DatasetTrialConfidence = NaN;
LossInformation.DatasetTaskConfidence  = NaN;
LossInformation.Confidence_Beta        = 1.0;
LossInformation.Confidence_Beta_Settle = 1.0;
LossInformation.Prior_Loss_Confidence  = 1.0;
LossInformation.Loss_Confidence_PerType = [1, 1, 1];

% Loss inputs to cgg_getConfidenceLossInformation — pass NaN-arrays of the
% right shape so the function's sum() over ValidClassificationIndices works
% but the Loss_Confidence sum contributes nothing per type. The Beta update
% only depends on the input confidences, not the loss values.
loss_arr_nan = dlarray(NaN(1, K));

states = cell(NumBatches, 1);
for bidx = 1:NumBatches
    b = batches{bidx};
    [~, LossInformation] = cgg_getConfidenceLossInformation( ...
        LossInformation, b.trial_dl, b.task_dl, ...
        loss_arr_nan, loss_arr_nan, loss_arr_nan, ...
        ValidClassificationIndices, BatchFraction);

    % Capture the post-update state for this batch.
    states{bidx} = struct( ...
        'confidence_beta',         LossInformation.Confidence_Beta, ...
        'dataset_total_confidence', LossInformation.DatasetTotalConfidence, ...
        'dataset_trial_confidence', LossInformation.DatasetTrialConfidence, ...
        'dataset_task_confidence',  LossInformation.DatasetTaskConfidence);
    fprintf('Batch %d: beta=%.6f  total_ema=%.6f  trial_ema=%.6f  task_ema=%.6f\n', ...
        bidx, states{bidx}.confidence_beta, ...
        states{bidx}.dataset_total_confidence, ...
        states{bidx}.dataset_trial_confidence, ...
        states{bidx}.dataset_task_confidence);
end

% ───────────────────────── Save fixture ─────────────────────────
fixture = struct( ...
    'K', K, 'B', B, 'T', T, ...
    'batch_fraction', BatchFraction, ...
    'num_batches', NumBatches, ...
    'beta_initial', 1.0, ...
    'beta_target', 0.5, ...
    'beta_difference_rate', 1.0, ...
    'beta_min', 0.1, ...
    'beta_max', 10.0);
for bidx = 1:NumBatches
    fixture.(sprintf('batch_%d_inputs', bidx)) = struct( ...
        'trial_in', batches{bidx}.trial_in, ...
        'task_in',  batches{bidx}.task_in);
    fixture.(sprintf('batch_%d_state',  bidx)) = states{bidx};
end

outDir = fullfile(projDir, 'tests', 'fixtures', 'golden_weights');
if ~isfolder(outDir); mkdir(outDir); end
outPath = fullfile(outDir, 'confidence_beta_t2.mat');
save(outPath, '-struct', 'fixture', '-v7');
fprintf('Wrote %s\n', outPath);
