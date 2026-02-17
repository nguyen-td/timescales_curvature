%% Setup
clear all
clc

seed = 3;
rng(seed)

%% Define random processes
n_frames = 11;         % number of video frames
frame_duration = 0.2;  % duration over with a single frame was shown (seconds)
n_trials = 1000;          % number of trials

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = 0.001;                       % simulation time steps (in s)
x = linspace(0, T, T / dt);       % dummy data points

% parameters of fast population
n_neurons = 100;    % number of neurons within the population
rho_f = 20;         % variance of tuning curves
tau_f = 0.9096;       % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = 0.1;  % [min, max] variance of gain
tau_g = 0.02; % time scale of gain (seconds)
q_g = 2;      % power law exponent of gain covariance function

%% Simulate neural responses
% generate tuning curves, gains, and rate functions
% [tuning_curves, gain, lambda] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f, rho_g, tau_g, q_g);

% %% Plot curves

% % plot tuning curves, gain and average rate separately
% itrial = 11;
% n_plots = 5; % number of neurons to plot side by side
% fig = figure;
% set(fig, 'Position', [0, 100, 1400, 700]); 
% set(gca, 'FontSize', 13)
% 
% for it = 1:n_plots
%     subplot(3, n_plots, it) % plot tuning curves
%     plot(x, squeeze(tuning_curves(it, :, itrial)), 'Color', 'black', 'LineWidth', 2)
%     ylim([min(tuning_curves, [], 'all') max(tuning_curves, [], "all")])
%     ylabel('f(s,t) (spikes/second)')
%     xlabel('Time (s)')
%     title(['Neuron ' num2str(it)])
% end
% lgd_tuning = legend('Stimulus-induced firing rate', 'Location', 'northwest', 'IconColumnWidth', 10);
% lgd_tuning.IconColumnWidth = 15;
% lgd_tuning.Box = 'off';
% 
% for ig = 1:n_plots
%     subplot(3, n_plots, ig + n_plots) % plot gains
%     plot(x, squeeze(gain(ig, :, itrial)), 'LineWidth', 2)
%     ylim([min(gain, [], 'all') max(gain, [], "all")])
%     ylabel('g(t) (spikes/second)')
%     xlabel('Time (s)')
% end
% lgd_gain = legend('Gain', 'Location', 'northwest', 'IconColumnWidth', 10);
% lgd_gain.IconColumnWidth = 15;
% lgd_gain.Box = 'off';
% 
% for il = 1:n_plots
%     subplot(3, n_plots, il + 2*n_plots) % plot gains
%     plot(x, squeeze(lambda(il, :, itrial)), 'LineWidth', 2)
%     ylim([min(lambda, [], 'all') max(lambda, [], "all")])
%     ylabel('\lambda(t) (spikes/second)')
%     xlabel('Time (s)')
% end
% lgd_lambda = legend('Modulated firing rate', 'Location', 'northwest', 'IconColumnWidth', 10);
% lgd_lambda.IconColumnWidth = 15;
% lgd_lambda.Box = 'off';
% 
% sgtitle(['Trial ' num2str(itrial)])

%% Compute curvature

% generate tuning curves, gains, and rate functions
mean_frange = [5, 50]; % determine the range of the firing rate, can be used to force realistic ranges
[tuning_curves, gain, lambda, K_g] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f, rho_g, tau_g, q_g, mean_frange);

clear S
S.tuning_curves = tuning_curves;
S.gain = gain;
S.lambda = lambda;

S.n_frames = n_frames;
S.frame_duration = frame_duration;
S.n_trials = n_trials;
S.T = T;
S.dt = dt;
S.x = x;
S.n_neuron = n_neurons;
S.rho_f = rho_f;
S.tau_f = tau_f;
S.rho_g = rho_g;
S.tau_g = tau_g;
S.q_g = q_g;

file_number = 0;
% save(fullfile('data', ['sim_' sprintf('%04d', file_number) '.mat']), 'S')
save(fullfile('data', ['sim_' sprintf('%04d', file_number) '_1k_trials.mat']), 'S', '-v7.3')