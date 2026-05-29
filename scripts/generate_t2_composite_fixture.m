% scripts/generate_t2_composite_fixture.m
%
% Milestone B composite T2 parity fixture: GRU Encoder + FC Bottleneck +
% Deep LSTM Classifier (per-dim), end to end.
%
% Builds the chain with the real MATLAB constructors
% (cgg_constructSimpleCoder for the encoder, cgg_generateLayersForClassifier
% for the per-dim classifier), deterministically initializes everything,
% forwards a fixed input, and saves all weights + input + per-dim
% pre-softmax logits to
% tests/fixtures/golden_weights/composite_t2_gru_deeplstm.mat.
%
% The Python parity test transplants every weight into an
% EncoderClassifierComposite (SimpleSequenceEncoder + LinearBottleneck +
% DeepLSTMClassifier) and asserts the per-dim logits match within fp32.
%
% Weight field naming: "<layer>__<param>" with '-' replaced by '_' so the
% names are valid MATLAB struct fields AND deterministically reproducible
% on the Python side (e.g. LSTM_Dim_1_Layer-Out -> LSTM_Dim_1_Layer_Out).

clear; close all;

% ───────────────────────── Path setup ─────────────────────────
addpath(fileparts(mfilename('fullpath')));   % put ndd_add_matlab_paths on path
ndd_add_matlab_paths();
thisDir = fileparts(mfilename('fullpath'));
projDir = fullfile(thisDir, '..');

% ───────────────────────── Configuration ─────────────────────────
InFeatures           = 3;
HiddenSizes          = [4, 2];   % encoder GRU stack
BottleNeckSize       = 3;        % bottleneck FC output
ClassifierHiddenSize = [4, 2];   % per-dim LSTM stack
NumClasses           = [3, 2];   % 2 output dimensions
NumTimesteps         = 10;
NumTrials            = 2;
Seed                 = 7;

% ───────────────────────── Encoder + bottleneck ─────────────────────────
encBlocks = cgg_constructSimpleCoder(HiddenSizes, 'Coder','Encoder', ...
    'Dropout',0, 'WantNormalization',false, 'Transform','GRU', 'Activation','');
encLayers = [sequenceInputLayer(InFeatures,'Name','Input_Encoder'); ...
             encBlocks; ...
             fullyConnectedLayer(BottleNeckSize,'Name','fc_OUT_BottleNeck', ...
                 'WeightsInitializer','he')];
EncNet = dlnetwork(layerGraph(encLayers));

rng(Seed,'twister');
X = randn(InFeatures, NumTimesteps, NumTrials);
X_dl = dlarray(X, 'CTB');

rng(Seed,'twister');
EncNet = initialize(EncNet, X_dl);
Enc_out = predict(EncNet, X_dl);          % CBT [BottleNeckSize, B, T]

% ───────────────────────── Per-dim classifier ─────────────────────────
ClsLayers = cgg_generateLayersForClassifier(NumClasses, ...
    'LossType','Classification', 'NetworkType','LSTM', ...
    'DropoutPercent',0.5, 'ClassifierHiddenSize',ClassifierHiddenSize, ...
    'MultipleInstanceLearningType','None', 'ConfidenceType','None');

NumDims = numel(NumClasses);
weights = struct();

% Encoder + bottleneck learnables.
encL = EncNet.Learnables;
for k = 1:height(encL)
    fname = local_fieldname(char(encL.Layer{k}), char(encL.Parameter{k}));
    weights.(fname) = extractdata(encL.Value{k});
end

% Per-dim classifier learnables + pre-softmax logits.
expected = struct();
for d = 1:NumDims
    lg = ClsLayers{d};
    lg = removeLayers(lg, sprintf('softmax_Tuning_Dim_%d', d));   % read pre-softmax fc output
    lg = addLayers(lg, sequenceInputLayer(BottleNeckSize,'Name','clsin'));
    lg = connectLayers(lg, 'clsin', sprintf('LSTM_Dim_%d_Layer-1', d));
    net = dlnetwork(lg);
    rng(100 + d, 'twister');
    net = initialize(net);

    L = net.Learnables;
    for k = 1:height(L)
        fname = local_fieldname(char(L.Layer{k}), char(L.Parameter{k}));
        weights.(fname) = extractdata(L.Value{k});
    end

    y = predict(net, Enc_out);            % CBT [NumClasses(d), B, T]
    expected.(sprintf('logits_%d', d)) = extractdata(y);
end

% ───────────────────────── Save fixture ─────────────────────────
fixture = struct( ...
    'in_features',          InFeatures, ...
    'hidden_sizes',         HiddenSizes, ...
    'bottleneck_size',      BottleNeckSize, ...
    'classifier_hidden_size', ClassifierHiddenSize, ...
    'num_classes',          NumClasses, ...
    'num_dims',             NumDims, ...
    'num_timesteps',        NumTimesteps, ...
    'num_trials',           NumTrials, ...
    'seed',                 Seed, ...
    'input',                X, ...
    'input_format',         'CTB', ...
    'logits_format',        'CBT', ...
    'weights',              weights, ...
    'expected',             expected);

outDir = fullfile(projDir, 'tests', 'fixtures', 'golden_weights');
if ~isfolder(outDir); mkdir(outDir); end
outPath = fullfile(outDir, 'composite_t2_gru_deeplstm.mat');
save(outPath, '-struct', 'fixture', '-v7');

fprintf('Wrote %s\n', outPath);
fprintf('  encoder+bottleneck output (CBT): [%s]\n', num2str(size(Enc_out)));
for d = 1:NumDims
    fprintf('  dim %d logits (CBT): [%s]\n', d, ...
        num2str(size(expected.(sprintf('logits_%d', d)))));
end
fprintf('  weight fields: %d\n', numel(fieldnames(weights)));

% ───────────────────────── Helper ─────────────────────────
function name = local_fieldname(layer, param)
    % Build a deterministic struct field name "<layer>__<param>" with
    % '-' replaced by '_' (the only non-field-safe char in our layer names).
    raw = [layer '__' param];
    name = strrep(raw, '-', '_');
end
