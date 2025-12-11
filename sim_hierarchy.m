%% Setup
clear all
clc

seed = 3;
rng(seed)

%% Simulate hierarchy of neural populations

n_frames = 11;        % number of video frames
frame_duration = 0.2; % duration over with a single frame was shown (seconds)
n_trials = 1000;         % number of trials

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = 0.001;                       % simulation time steps (in s)
x = linspace(0, T, T / dt);       % dummy data points

% parameters of fast population
n_neurons = 20;                  % number of neurons within the population
rho_f = 3;                       % variance of tuning curves
tau_f = linspace(0.001, 2, 100);  % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = 0.1;           % [min, max] variance of gain
tau_g = [0.002, 0.02]; % time scale of gain (seconds)
q_g = 2;               % power law exponent of gain covariance function
% tau_g = 0.02;          % time scale of gain (seconds)
% q_g = [0.5, 2];        % power law exponent of gain covariance function

%% Simulate neural responses

% find suitable bin sizes
clear bin_sizes_all
count = 1;
for bin = 0:dt:T
    if mod(numel(x), bin / dt) == 0
        bin_sizes_all(count) = bin;
        count = count + 1;
    end
end

% loop over hierarchy of neural populations
c_slow_gain = zeros(numel(bin_sizes_all), numel(tau_f));
c_fast_gain = zeros(numel(bin_sizes_all), numel(tau_f));

for itau = 1:numel(tau_f)
    [tuning_curves_fast_gain, ~, lambda_fast_gain] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g(1), q_g);
    [tuning_curves_slow_gain, ~, lambda_slow_gain] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g(2), q_g);
    % [tuning_curves_fast_gain, ~, ~] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g, q_g(1));
    % [tuning_curves_slow_gain, ~, ~] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g, q_g(2));

    % loop over different bin sizes
    for ibin = 1:numel(bin_sizes_all)
        [mean_lambda_fast_gain, ~, ~, var_gain_fast_gain] = mean_var_CMP(tuning_curves_fast_gain, rho_g, dt, T, bin_sizes_all(ibin), tau_g(1), q_g);
        [mean_lambda_slow_gain, ~, ~, var_gain_slow_gain] = mean_var_CMP(tuning_curves_slow_gain, rho_g, dt, T, bin_sizes_all(ibin), tau_g(2), q_g);
        % [mean_lambda_fast_gain, ~, ~, var_gain_fast_gain] = mean_var_CMP(tuning_curves_fast_gain, rho_g, dt, T, bin_sizes_all(ibin), tau_g, q_g(1));
        % [mean_lambda_slow_gain, ~, ~, var_gain_slow_gain] = mean_var_CMP(tuning_curves_slow_gain, rho_g, dt, T, bin_sizes_all(ibin), tau_g, q_g(2));

        % compute global curvature (in degrees)
        c_fast_gain(ibin, itau) = rad2deg(mean(compute_curvature(var_gain_fast_gain, mean_lambda_fast_gain)));
        c_slow_gain(ibin, itau) = rad2deg(mean(compute_curvature(var_gain_slow_gain, mean_lambda_slow_gain)));
    end
end

%% Plot results

n_rows = 6;
n_cols = floor(numel(bin_sizes_all) / n_rows);
max_curvature = max([c_fast_gain, c_slow_gain], [], 'all');

figure(1)
tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'none');
set(gca, 'FontSize', 13)

for ibin = 1:numel(bin_sizes_all)
    nexttile
    scatter(tau_f, c_slow_gain(ibin, :), 10, 'filled')
    hold on;
    scatter(tau_f, c_fast_gain(ibin, :), 10, 'filled')
    title(['\Deltat = ' num2str(bin_sizes_all(ibin)) 's'])
    xlabel('Time scale of f(s,t) (s)')
    ylabel('Curvature (deg)')

    legend('Slow gain', 'Fast gain')
end