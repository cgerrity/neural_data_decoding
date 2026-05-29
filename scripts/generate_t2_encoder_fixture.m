% scripts/generate_t2_encoder_fixture.m
%
% Milestone B T2 single-step forward-pass parity fixture generator.
%
% Builds a small Simple-branch GRU encoder via cgg_constructSimpleCoder,
% deterministically initializes it, runs `predict()` on a fixed input,
% and saves all weights + input + expected output to
% tests/fixtures/encoder_t2_gru_simple.mat.
%
% The Python test
% (tests/parity/test_t2_encoder_forward_parity.py) loads this fixture,
% transplants the weights into a matching SimpleSequenceEncoder, runs
% the same input through the Python forward pass, and asserts the
% outputs are equal within fp32 tolerance.
%
% Run from the neural_data_decoding directory:
%     >> run('scripts/generate_t2_encoder_fixture.m')
% or from a shell:
%     $ matlab -batch "run('scripts/generate_t2_encoder_fixture.m')"

clear; close all;

% ───────────────────────── Path setup ─────────────────────────
addpath(fileparts(mfilename('fullpath')));   % put ndd_add_matlab_paths on path
ndd_add_matlab_paths();
thisDir = fileparts(mfilename('fullpath'));
projDir = fullfile(thisDir, '..');           % neural_data_decoding/

% ───────────────────────── Test configuration ─────────────────────────
InFeatures   = 3;
HiddenSizes  = [4, 2];
NumTimesteps = 10;
NumTrials    = 2;
Seed         = 42;

% ───────────────────────── Build the encoder ─────────────────────────
% Use the same constructor the production pipeline uses.
encoderBlocks = cgg_constructSimpleCoder(HiddenSizes, ...
    'Coder', 'Encoder', ...
    'Dropout', 0, ...                % deterministic at predict() time anyway
    'WantNormalization', false, ...
    'Transform', 'GRU', ...
    'Activation', '');

inputLayer = sequenceInputLayer(InFeatures, 'Name', 'Input_Encoder');
fullLayers = [inputLayer; encoderBlocks];
net = dlnetwork(layerGraph(fullLayers));

% ───────────────────────── Generate deterministic input ─────────────────────────
rng(Seed, 'twister');
X = randn(InFeatures, NumTimesteps, NumTrials);
X_dl = dlarray(X, 'CTB');

% ───────────────────────── Initialize weights deterministically ─────────────────────────
rng(Seed, 'twister');
net = initialize(net, X_dl);

% ───────────────────────── Verify ResetGateMode parity ─────────────────────────
% PyTorch's nn.GRU implements the "after-multiplication" formulation:
%   n_t = tanh(W_in x + b_in + r * (W_hn h + b_hn))
% MATLAB gruLayer's default is also 'after-multiplication'. If a future
% MATLAB change moves the default, this assertion catches it before the
% Python test silently produces wrong numbers.
for k = 1:numel(net.Layers)
    L = net.Layers(k);
    if isa(L, 'nnet.cnn.layer.GRULayer')
        assert(strcmp(L.ResetGateMode, 'after-multiplication'), ...
            'GRU layer "%s" has ResetGateMode="%s"; expected "after-multiplication".', ...
            L.Name, L.ResetGateMode);
    end
end

% ───────────────────────── Forward pass ─────────────────────────
Y_dl = predict(net, X_dl);
Y = extractdata(Y_dl);
output_format = char(Y_dl.dims);

% ───────────────────────── Extract weights into a flat struct ─────────────────────────
% Each (Layer, Parameter) pair becomes one field named
% "<layer>__<parameter>" (struct field names can't contain dots).
weights = struct();
learnables = net.Learnables;
for k = 1:height(learnables)
    layerName = learnables.Layer{k};
    paramName = learnables.Parameter{k};
    fieldName = matlab.lang.makeValidName([char(layerName) '__' char(paramName)]);
    weights.(fieldName) = extractdata(learnables.Value{k});
end

% ───────────────────────── Layer-order metadata ─────────────────────────
layer_names   = {};
layer_classes = {};
layer_sizes   = [];
for k = 1:numel(net.Layers)
    L = net.Layers(k);
    layer_names{end+1}   = L.Name; %#ok<AGROW>
    layer_classes{end+1} = class(L); %#ok<AGROW>
    if isa(L, 'nnet.cnn.layer.GRULayer') || isa(L, 'nnet.cnn.layer.LSTMLayer')
        layer_sizes(end+1) = L.NumHiddenUnits; %#ok<AGROW>
    elseif isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
        layer_sizes(end+1) = L.OutputSize; %#ok<AGROW>
    else
        layer_sizes(end+1) = -1; %#ok<AGROW>
    end
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
    'weights',         weights, ...
    'layer_names',     {layer_names}, ...
    'layer_classes',   {layer_classes}, ...
    'layer_sizes',     layer_sizes);

outDir  = fullfile(projDir, 'tests', 'fixtures', 'golden_weights');
if ~isfolder(outDir); mkdir(outDir); end
outPath = fullfile(outDir, 'encoder_t2_gru_simple.mat');
save(outPath, '-struct', 'fixture', '-v7');

fprintf('Wrote %s\n', outPath);
fprintf('  input  (CTB): shape [%s]\n', num2str(size(X)));
fprintf('  output (%s): shape [%s]\n', output_format, num2str(size(Y)));
fprintf('  weights: %d fields (%s)\n', numel(fieldnames(weights)), ...
    strjoin(fieldnames(weights), ', '));
