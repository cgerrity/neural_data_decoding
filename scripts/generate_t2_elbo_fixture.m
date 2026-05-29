% scripts/generate_t2_elbo_fixture.m
%
% Milestone C VAE-core T2 parity fixture: cgg_lossELBO_v2 + cgg_samplingLayer.
%
% Saves a known (Y, T, mu, logSigmaSq) with hand-placed NaNs in T, the
% three ELBO outputs (reconstruction, KL, per-channel reconstruction), and
% a sampling-layer predict() round-trip, to
% tests/fixtures/golden_weights/elbo_t2.mat.
%
% The Python parity test
% (tests/parity/test_t2_elbo_parity.py) reproduces each value with the
% masked_mse_reconstruction_loss / kl_divergence_loss kernels and the
% SamplingLayer, asserting fp32 agreement. This nails the NaN-mask
% batch-size normalization (the highest-risk silent-parity point per
% Critical Note #38).
%
% Reconstruction data is SSCTB [S1, S2, C, T, B] so the per-channel loop
% (which slices the C = 3rd dim) is exercised. Latent stats are CBT.

clear; close all;

% ───────────────────────── Path setup ─────────────────────────
addpath(fileparts(mfilename('fullpath')));   % put ndd_add_matlab_paths on path
ndd_add_matlab_paths();
thisDir = fileparts(mfilename('fullpath'));
projDir = fullfile(thisDir, '..');

rng(11, 'twister');

% ───────────────────────── Reconstruction tensors (SSCTB) ─────────────────────────
S1 = 2; S2 = 2; C = 3; T = 4; B = 2;
Y = randn(S1, S2, C, T, B);
T_target = randn(S1, S2, C, T, B);
% Hand-place NaNs at two known positions (simulating removed channels).
T_target(1, 1, 2, 1, 1) = NaN;
T_target(2, 2, 3, 4, 2) = NaN;

Y_dl = dlarray(Y, 'SSCTB');
T_dl = dlarray(T_target, 'SSCTB');

% ───────────────────────── Latent stats (CBT) ─────────────────────────
Lat = 3;
mu = randn(Lat, B, T);
logSigmaSq = randn(Lat, B, T);
mu_dl = dlarray(mu, 'CBT');
logSigmaSq_dl = dlarray(logSigmaSq, 'CBT');

% ───────────────────────── ELBO ─────────────────────────
[loss_Reconstruction, loss_KL, loss_Reconstruction_perchannel] = ...
    cgg_lossELBO_v2(Y_dl, T_dl, mu_dl, logSigmaSq_dl);

loss_Reconstruction      = double(extractdata(loss_Reconstruction));
loss_KL                  = double(extractdata(loss_KL));
loss_Recon_perchannel    = double(loss_Reconstruction_perchannel(:));   % already extracted

fprintf('loss_Reconstruction = %.10g\n', loss_Reconstruction);
fprintf('loss_KL             = %.10g\n', loss_KL);
fprintf('per-channel         = [%s]\n', num2str(loss_Recon_perchannel'));

% ───────────────────────── Sampling layer predict() round-trip ─────────────────────────
% Concatenate [mu; logSigmaSq] along the channel dim (CBT → dim 1).
X_sampling = cat(1, mu, logSigmaSq);          % [2*Lat, B, T]
X_sampling_dl = dlarray(X_sampling, 'CBT');
layer = cgg_samplingLayer('DataFormat', 'CBT');
[Z_pred, mu_pred, logvar_pred] = predict(layer, X_sampling_dl);
Z_pred      = double(extractdata(Z_pred));
mu_pred     = double(extractdata(mu_pred));
logvar_pred = double(extractdata(logvar_pred));
fprintf('sampling predict: max|Z - mu| = %.3e (should be ~0, deterministic)\n', ...
    max(abs(Z_pred(:) - mu_pred(:))));

% ───────────────────────── Save fixture ─────────────────────────
fixture = struct( ...
    'Y',                            Y, ...
    'T_target',                     T_target, ...
    'recon_format',                 'SSCTB', ...
    'recon_batch_dim_matlab',       5, ...   % 1-indexed B position
    'recon_channel_dim_matlab',     3, ...   % 1-indexed C position
    'mu',                           mu, ...
    'logSigmaSq',                   logSigmaSq, ...
    'latent_format',                'CBT', ...
    'latent_channel_dim_matlab',    1, ...
    'loss_Reconstruction',          loss_Reconstruction, ...
    'loss_KL',                      loss_KL, ...
    'loss_Reconstruction_perchannel', loss_Recon_perchannel, ...
    'X_sampling',                   X_sampling, ...
    'Z_pred',                       Z_pred, ...
    'mu_pred',                      mu_pred, ...
    'logvar_pred',                  logvar_pred);

outDir = fullfile(projDir, 'tests', 'fixtures', 'golden_weights');
if ~isfolder(outDir); mkdir(outDir); end
outPath = fullfile(outDir, 'elbo_t2.mat');
save(outPath, '-struct', 'fixture', '-v7');
fprintf('Wrote %s\n', outPath);
