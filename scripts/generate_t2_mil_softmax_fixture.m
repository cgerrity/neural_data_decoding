% scripts/generate_t2_mil_softmax_fixture.m
%
% Milestone C MIL-pooling T2 parity fixture: cgg_softmaxLayer.
%
% cgg_softmaxLayer computes a softmax JOINTLY over all axes whose dlarray
% format tag is in the requested SoftmaxFormat string (Critical Note #10:
% "MIL pooling is multi-axis softmax across Space-Channel-Time"). This
% fixture exercises three layouts so the Python MILSoftmaxLayer's
% from_formats() axis mapping (find(ismember(...))) is pinned exactly:
%
%   1. CBT   input, format 'SCT'  -> softmax over (C, T)        [dims 1,3]
%   2. CBT   input, format 'C'    -> softmax over (C)           [dim  1]
%   3. SSCTB input, format 'SCT'  -> softmax over (S1,S2,C,T)   [dims 1-4]
%
% Saves inputs + outputs to tests/fixtures/golden_weights/mil_softmax_t2.mat.
% The Python parity test (tests/parity/test_t2_mil_softmax_parity.py)
% reproduces each output with MILSoftmaxLayer and asserts fp32 agreement.

clear; close all;

% ───────────────────────── Path setup ─────────────────────────
thisDir   = fileparts(mfilename('fullpath'));
projDir   = fullfile(thisDir, '..');
parentDir = fullfile(projDir, '..');
for sub = {'Processing_Functions_cgg','FLU_Process_scripts_LT', ...
           'LoopUtil','exp-utils-cjt-4','External_Functions','YAMLMatlab_0.4.3'}
    p = fullfile(parentDir, sub{1});
    if isfolder(p); addpath(genpath(p)); end
end

rng(23, 'twister');

% NOTE: a formatted dlarray is canonicalized by MATLAB on extractdata, so
% the stored dimension order may differ from the order at construction
% (e.g. 'SSCTB' comes back as 'SSCBT'). extract_as() (local function below)
% permutes the extracted data back into a requested label order so the saved
% input and output share one unambiguous layout for the Python parity test.

% ───────────────────────── Case 1: CBT, 'SCT' ─────────────────────────
C = 3; B = 2; T = 4;
X_cbt = randn(C, B, T);
X_cbt_dl = dlarray(X_cbt, 'CBT');
layer_sct = cgg_softmaxLayer('SCT');
Z_cbt_sct = extract_as(predict(layer_sct, X_cbt_dl), 'CBT');
% Joint softmax over (C, T) -> for each batch element, sum over C&T == 1.
chk1 = squeeze(sum(sum(Z_cbt_sct, 1), 3));   % [B x 1]
fprintf('Case1 CBT/SCT: max|sum_CT - 1| = %.3e\n', max(abs(chk1(:) - 1)));

% ───────────────────────── Case 2: CBT, 'C' ─────────────────────────
layer_c = cgg_softmaxLayer('C');
Z_cbt_c = extract_as(predict(layer_c, X_cbt_dl), 'CBT');
% Softmax over C only -> sum over C == 1 for each (B, T).
chk2 = squeeze(sum(Z_cbt_c, 1));             % [B x T]
fprintf('Case2 CBT/C:   max|sum_C - 1|  = %.3e\n', max(abs(chk2(:) - 1)));

% ───────────────────────── Case 3: SSCTB, 'SCT' ─────────────────────────
S1 = 2; S2 = 2; Cs = 3; Ts = 4; Bs = 2;
X_ssctb = randn(S1, S2, Cs, Ts, Bs);
X_ssctb_dl = dlarray(X_ssctb, 'SSCTB');
Z_ssctb_sct = extract_as(predict(layer_sct, X_ssctb_dl), 'SSCTB');
% Joint softmax over (S1,S2,C,T) -> for each batch element, sum == 1.
chk3 = squeeze(sum(sum(sum(sum(Z_ssctb_sct, 1), 2), 3), 4));  % [B x 1]
fprintf('Case3 SSCTB/SCT: max|sum_SSCT - 1| = %.3e\n', max(abs(chk3(:) - 1)));

% ───────────────────────── Save fixture ─────────────────────────
fixture = struct( ...
    'X_cbt',        X_cbt, ...
    'X_cbt_format', 'CBT', ...
    'Z_cbt_sct',    Z_cbt_sct, ...
    'Z_cbt_c',      Z_cbt_c, ...
    'X_ssctb',      X_ssctb, ...
    'X_ssctb_format','SSCTB', ...
    'Z_ssctb_sct',  Z_ssctb_sct);

outDir = fullfile(projDir, 'tests', 'fixtures', 'golden_weights');
if ~isfolder(outDir); mkdir(outDir); end
outPath = fullfile(outDir, 'mil_softmax_t2.mat');
save(outPath, '-struct', 'fixture', '-v7');
fprintf('Wrote %s\n', outPath);

% ───────────────────────── Local functions ─────────────────────────
function Zout = extract_as(Z_dl, targetFmt)
    % Extract a formatted dlarray into a requested label order, undoing
    % MATLAB's canonical-order reshuffle. Repeated spatial labels ('S')
    % are matched in left-to-right order to preserve their relative order.
    curFmt = char(dims(Z_dl));
    raw = double(extractdata(Z_dl));
    if isempty(curFmt)
        Zout = raw;
        return;
    end
    used = false(1, numel(curFmt));
    perm = zeros(1, numel(targetFmt));
    for i = 1:numel(targetFmt)
        idx = find(curFmt == targetFmt(i) & ~used, 1);
        used(idx) = true;
        perm(i) = idx;
    end
    Zout = permute(raw, perm);
end
