% scripts/generate_t2_dynamic_schedule_interpolator_fixture.m
%
% Milestone C #5 — piecewise-anneal interpolator T2 parity fixture.
%
% Probes ``cgg_calculateDynamicValue`` (composed with the inner
% ``cgg_annealWeight`` helper) across multiple cases. Each case saves the
% inputs (base, epoch_points, magnitude_points) and the per-epoch outputs
% over a dense epoch grid so the Python parity test can pin every value.
%
% Cases:
%   A. Increasing 2-waypoint schedule       — basic ramp + clamps.
%   B. Decreasing 2-waypoint schedule       — verifies direction.
%   C. 3-waypoint with constant first       — pins inner-waypoint
%      segment ([0,50,75], [1e-2,1e-2,1.0])    discontinuity from the
%                                              (epoch-1) off-by-one.
%   D. Real "Soft Three-Stage Curriculum -  — every parameter (weights,
%      Shortened" regime                       freeze, augmentation) at
%                                              all 50 epochs (covers the
%                                              schedule library too).
%   E. Single waypoint                      — left/right clamp edge.
%   F. Wide segment ([10,40])               — multiple in-segment epochs
%                                              to see the linear ramp.
%
% The off-by-one quirk: at every internal waypoint the in-segment ramp
% reaches only (span - 1) / span of the way to the next magnitude, then
% the next segment's leading edge (or the right-clamp) snaps to the
% exact magnitude. Case C is designed to exhibit this clearly because
% the segments share magnitude (1e-2 → 1e-2 then ramping to 1.0).

clear; close all;

% ───────────────────────── Path setup ─────────────────────────
addpath(fileparts(mfilename('fullpath')));
ndd_add_matlab_paths();
thisDir = fileparts(mfilename('fullpath'));
projDir = fullfile(thisDir, '..');

% ───────────────────────── Case A: increasing 2-waypoint ─────────────────────────
case_A.base             = 1.0;
case_A.epoch_points     = [10, 20];
case_A.magnitude_points = [0.1, 1.0];
case_A.epochs           = 1:30;
case_A.values           = arrayfun(@(e) cgg_calculateDynamicValue( ...
    case_A.base, case_A.epoch_points, case_A.magnitude_points, e), ...
    case_A.epochs);
fprintf('Case A: ramp [0.1→1.0] over [10,20], probed 1..30 (n=%d)\n', numel(case_A.epochs));

% ───────────────────────── Case B: decreasing 2-waypoint ─────────────────────────
case_B.base             = 2.0;       % nontrivial base to verify multiplication
case_B.epoch_points     = [20, 30];
case_B.magnitude_points = [1.0, 1e-2];
case_B.epochs           = 1:40;
case_B.values           = arrayfun(@(e) cgg_calculateDynamicValue( ...
    case_B.base, case_B.epoch_points, case_B.magnitude_points, e), ...
    case_B.epochs);
fprintf('Case B: ramp [1.0→1e-2] over [20,30] base=2.0, probed 1..40 (n=%d)\n', ...
    numel(case_B.epochs));

% ───────────────────────── Case C: 3-waypoint with constant first segment ─────────────────────────
case_C.base             = 1.0;
case_C.epoch_points     = [0, 50, 75];
case_C.magnitude_points = [1e-2, 1e-2, 1.0];
case_C.epochs           = 1:100;
case_C.values           = arrayfun(@(e) cgg_calculateDynamicValue( ...
    case_C.base, case_C.epoch_points, case_C.magnitude_points, e), ...
    case_C.epochs);
fprintf('Case C: 3-waypoint [1e-2,1e-2,1.0] over [0,50,75], probed 1..100 (n=%d)\n', ...
    numel(case_C.epochs));

% ───────────────────────── Case D: real "Soft Three-Stage Curriculum - Shortened" regime ─────────────────────────
% Probes every parameter in the regime at all 50 epochs. Saves both the
% raw regime structs (so the Python schedule-library port can be pinned
% to the same waypoints) AND the per-epoch values for each parameter.
[D_aug, D_weights, D_freeze, D_desc] = PARAMETERS_cgg_selectDynamicParameters( ...
    'Soft Three-Stage Curriculum - Shortened');

case_D = struct();
case_D.regime_name       = 'Soft Three-Stage Curriculum - Shortened';
case_D.regime_description = char(D_desc);
case_D.dynamic_weighting = D_weights;
case_D.dynamic_augmentation = D_aug;
case_D.dynamic_freezing  = D_freeze;
case_D.epochs            = 1:50;

% Weights — each parameter has its own schedule (nested struct).
weight_fields = fieldnames(D_weights);
weight_values = struct();
for k = 1:numel(weight_fields)
    fld = weight_fields{k};
    ep  = D_weights.(fld).EpochPoints;
    mag = D_weights.(fld).MagnitudePoints;
    weight_values.(fld) = arrayfun(@(e) cgg_calculateDynamicValue( ...
        1.0, ep, mag, e), case_D.epochs);
end
case_D.weight_values_base1 = weight_values;
fprintf('Case D weights: %s\n', strjoin(weight_fields, ', '));

% Freezing — same nested per-parameter structure.
freeze_fields = fieldnames(D_freeze);
freeze_values = struct();
for k = 1:numel(freeze_fields)
    fld = freeze_fields{k};
    ep  = D_freeze.(fld).EpochPoints;
    mag = D_freeze.(fld).MagnitudePoints;
    freeze_values.(fld) = arrayfun(@(e) cgg_calculateDynamicValue( ...
        1.0, ep, mag, e), case_D.epochs);
end
case_D.freeze_values_base1 = freeze_values;
fprintf('Case D freeze: %s\n', strjoin(freeze_fields, ', '));

% Augmentation — flat (single shared schedule); applies uniformly to every
% augmentation parameter the LoadSchedule manages.
case_D.aug_epoch_points = D_aug.EpochPoints;
case_D.aug_magnitude_points = D_aug.MagnitudePoints;
case_D.aug_values_base1 = arrayfun(@(e) cgg_calculateDynamicValue( ...
    1.0, D_aug.EpochPoints, D_aug.MagnitudePoints, e), case_D.epochs);
fprintf('Case D augmentation: shared schedule, waypoints [%s]\n', ...
    num2str(D_aug.EpochPoints));

% ───────────────────────── Case E: single waypoint ─────────────────────────
case_E.base             = 1.0;
case_E.epoch_points     = 10;
case_E.magnitude_points = 0.5;
case_E.epochs           = 1:20;
case_E.values           = arrayfun(@(e) cgg_calculateDynamicValue( ...
    case_E.base, case_E.epoch_points, case_E.magnitude_points, e), ...
    case_E.epochs);
fprintf('Case E: single waypoint at epoch 10 (magnitude 0.5), probed 1..20\n');

% ───────────────────────── Case F: wide segment ─────────────────────────
case_F.base             = 1.0;
case_F.epoch_points     = [10, 40];
case_F.magnitude_points = [0.0, 1.0];
case_F.epochs           = 1:50;
case_F.values           = arrayfun(@(e) cgg_calculateDynamicValue( ...
    case_F.base, case_F.epoch_points, case_F.magnitude_points, e), ...
    case_F.epochs);
fprintf('Case F: wide ramp [0→1] over [10,40] (span=30), probed 1..50\n');

% ───────────────────────── Save fixture ─────────────────────────
fixture = struct( ...
    'case_A', case_A, ...
    'case_B', case_B, ...
    'case_C', case_C, ...
    'case_D', case_D, ...
    'case_E', case_E, ...
    'case_F', case_F);

outDir = fullfile(projDir, 'tests', 'fixtures', 'golden_weights');
if ~isfolder(outDir); mkdir(outDir); end
outPath = fullfile(outDir, 'dynamic_schedule_interpolator_t2.mat');
save(outPath, '-struct', 'fixture', '-v7');
fprintf('Wrote %s\n', outPath);
