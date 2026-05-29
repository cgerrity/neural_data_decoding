% scripts/generate_t2_encoder_fixture_lstm.m
%
% LSTM counterpart of generate_t2_encoder_fixture.m. Builds a tiny
% 2-layer Simple LSTM via cgg_constructSimpleCoder, deterministically
% initializes it, runs predict() on a fixed input, saves all weights +
% input + expected output to
% tests/fixtures/golden_weights/encoder_t2_lstm_simple.mat.
%
% MATLAB lstmLayer and PyTorch nn.LSTM share gate ordering
% [input, forget, cell-candidate, output]. Unlike GRU there's no
% reset-gate subtlety on the candidate path, so the bias mapping is
% bias_ih_l0 = MATLAB Bias, bias_hh_l0 = 0 — same recipe as GRU but
% applied to all four gates uniformly.

clear; close all;

% ───────────────────────── Path setup ─────────────────────────
addpath(fileparts(mfilename('fullpath')));   % put ndd_add_matlab_paths on path
ndd_add_matlab_paths();
thisDir = fileparts(mfilename('fullpath'));
projDir = fullfile(thisDir, '..');

% ───────────────────────── Test configuration ─────────────────────────
InFeatures   = 3;
HiddenSizes  = [4, 2];
NumTimesteps = 10;
NumTrials    = 2;
Seed         = 43;     % different from GRU fixture's seed so the two
                       % parity tests can't accidentally cross-validate.

% ───────────────────────── Build the encoder ─────────────────────────
encoderBlocks = cgg_constructSimpleCoder(HiddenSizes, ...
    'Coder', 'Encoder', ...
    'Dropout', 0, ...
    'WantNormalization', false, ...
    'Transform', 'LSTM', ...
    'Activation', '');

inputLayer = sequenceInputLayer(InFeatures, 'Name', 'Input_Encoder');
net = dlnetwork(layerGraph([inputLayer; encoderBlocks]));

% ───────────────────────── Deterministic input + initialization ─────────────────────────
rng(Seed, 'twister');
X = randn(InFeatures, NumTimesteps, NumTrials);
X_dl = dlarray(X, 'CTB');

rng(Seed, 'twister');
net = initialize(net, X_dl);

% ───────────────────────── Forward pass ─────────────────────────
Y_dl = predict(net, X_dl);
Y = extractdata(Y_dl);
output_format = char(Y_dl.dims);

% ───────────────────────── Extract weights ─────────────────────────
weights = struct();
learnables = net.Learnables;
for k = 1:height(learnables)
    layerName = learnables.Layer{k};
    paramName = learnables.Parameter{k};
    fieldName = matlab.lang.makeValidName([char(layerName) '__' char(paramName)]);
    weights.(fieldName) = extractdata(learnables.Value{k});
end

% ───────────────────────── Save fixture ─────────────────────────
fixture = struct( ...
    'in_features',     InFeatures, ...
    'hidden_sizes',    HiddenSizes, ...
    'num_timesteps',   NumTimesteps, ...
    'num_trials',      NumTrials, ...
    'seed',            Seed, ...
    'input',           X, ...
    'input_format',    'CTB', ...
    'expected_output', Y, ...
    'output_format',   output_format, ...
    'weights',         weights);

outDir  = fullfile(projDir, 'tests', 'fixtures', 'golden_weights');
if ~isfolder(outDir); mkdir(outDir); end
outPath = fullfile(outDir, 'encoder_t2_lstm_simple.mat');
save(outPath, '-struct', 'fixture', '-v7');

fprintf('Wrote %s\n', outPath);
fprintf('  input  (CTB): shape [%s]\n', num2str(size(X)));
fprintf('  output (%s): shape [%s]\n', output_format, num2str(size(Y)));
fprintf('  weights: %d fields (%s)\n', numel(fieldnames(weights)), ...
    strjoin(fieldnames(weights), ', '));
