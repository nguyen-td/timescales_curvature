%% Setup
clear all
clc

seed = 30;
rng(seed)

%% Define random processes
n_frames = 11;           % number of video frames
frame_duration = 0.2;    % duration over with a single frame was shown (seconds)
n_trials = 100;          % number of trials

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = 0.001;                       % simulation time steps (in s)
x = linspace(0, T, T / dt);       % dummy data points

% parameters of fast population
n_neurons = 10;    % number of neurons within the population
rho_f = 1;         % variance of tuning curves
tau_f = 0.5;       % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = 0.1;  % [min, max] variance of gain
tau_g = 0.02;  % time scale of gain (seconds)
q_g = 2;      % power law exponent of gain covariance function

%% Simulate neural responses
[tuning_curves, gain, lambda, K_g] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f, rho_g, tau_g, q_g);

%% Plot curves

% plot tuning curves, gain and average rate separately
itrial = 14;
n_plots = 5; % number of neurons to plot side by side
fig = figure;
set(fig, 'Position', [0, 100, 1400, 700]); 
set(gca, 'FontSize', 13)

for it = 1:n_plots
    subplot(3, n_plots, it) % plot tuning curves
    plot(x, squeeze(tuning_curves(it, :, itrial)), 'Color', 'black', 'LineWidth', 1)
    ylim([min(tuning_curves, [], 'all') max(tuning_curves, [], "all")])
    ylabel('Firing rate (spikes/second)')
    xlabel('Time (s)')
    title(['Neuron ' num2str(it)])
end
lgd_tuning = legend('Stimulus-induced firing rate', 'Location', 'northwest', 'IconColumnWidth', 10);
lgd_tuning.IconColumnWidth = 15;
lgd_tuning.Box = 'off';

for ig = 1:n_plots
    subplot(3, n_plots, ig + n_plots) % plot gains
    plot(x, squeeze(gain(ig, :, itrial)), 'LineWidth', 1)
    % ylim([min(gain, [], 'all') max(gain, [], "all")])
    ylabel('Gain (spikes/second)')
    xlabel('Time (s)')
end
lgd_gain = legend('Gain', 'Location', 'northwest', 'IconColumnWidth', 10);
lgd_gain.IconColumnWidth = 15;
lgd_gain.Box = 'off';

for il = 1:n_plots
    subplot(3, n_plots, il + 2*n_plots) % plot gains
    plot(x, squeeze(lambda(il, :, itrial)), 'LineWidth', 1)
    ylim([min(lambda, [], 'all') max(lambda, [], "all")])
    ylabel('Modulated firing rate (spikes/second)')
    xlabel('Time (s)')
end
lgd_lambda = legend('Modulated firing rate', 'Location', 'northwest', 'IconColumnWidth', 10);
lgd_lambda.IconColumnWidth = 15;
lgd_lambda.Box = 'off';

sgtitle(['Trial ' num2str(itrial)])

% % compute gain and autocorrelation
% delta = linspace(-T, T, 2 * length(x) + 1);
% figure; plot(autocorr_g(delta, tau_g, q_g))
% figure; plot(x, gain)

%% Compute curvature
% [mean_GP, var_GP, kappa] = mean_var_CMP(tuning_curves, rho_g, dt, dt, tau_g, q_g);

% fano_factor = mean(var_GP ./ mean_GP, 'all');
% fano_factor = 1 + mean_GP .* (kappa - 1);
% disp(['Fano factor: ' num2str(fano_factor)])
% disp(['Kappa ' num2str(kappa)])

% % compute embedding on trial averages
% y = 2 ./ mean(var_process, 3) .* asinh(mean(var_process, 3) .* sqrt(mean(lambda, 3))); % is an approximation, was derived for constant gain
% c = compute_trajectory_pixel(reshape(y, [1, size(y, 1), size(y, 2)]));
% disp(['Average curvature: ' num2str(mean(rad2deg(c))) ' degrees'])