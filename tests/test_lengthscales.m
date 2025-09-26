%% Setup
clear all
clc

%% Define random process
n_frames = 11;           % number of video frames
frame_duration = 0.2;    % duration over with a single frame was shown (seconds)
n_trials = 1000;         % number of trials

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = [0.1, 0.002, 0.003];  % bin size (in s)
x = {};                           % time bins (unit: T / dt)
for i=1:length(dt)
    x{i} = linspace(0, T, T / dt(i));       
end

% parameters of tuning curves
n_neurons = 1;     % number of neurons within the population
rho_f = 2;         % variance of tuning curves
tau_f = 2;         % time scale of tuning curves

% parameters of gains
rho_g = 0.1;  % [min, max] variance of gain
tau_g = 0.2;  % time scale of gain (seconds)
q_g = 2;      % power law exponent of gain covariance function

%% Simulate neural responses
tuning_curves = {};
gain = {};
lambda = {};
K_g = {};

for i=1:length(dt)
    rng(42)
    % [tuning_curves_dt, gain_dt, lambda_dt, K_g_dt] = simulate_mod_poisson(n_trials, x{i}, n_neurons, rho_f, tau_f, rho_g, tau_g, q_g);
    [tuning_curves_dt, gain_dt, lambda_dt, ~] = compute_CMP(n_trials, x{i}, n_neurons, rho_f, tau_f, rho_g, tau_g, q_g)  ;  

    tuning_curves{i} = mean(tuning_curves_dt, 3); % save trial average
    gain{i} = mean(gain_dt, 3); % save trial average
    lambda{i} = mean(lambda_dt, 3);
    % K_g{i} = mean(K_g_dt, 1);
end

%% Plot

% plot tuning curves
figure(1); 
set(gca, 'FontSize', 13)
for i=1:length(dt)
    plot(x{i}, tuning_curves{i}, '-o', 'LineWidth', 1)
    hold on;
    xlabel('Time (s)')
    ylabel('Firing rate (spikes/s)')
end
title('Tuning curves')
legend(['dt = ' num2str(dt(1) * 1000) ' ms'], ['dt = ' num2str(dt(2) * 1000) ' ms'], ['dt = ' num2str(dt(3) * 1000) ' ms']);

% plot gains
figure(2); 
set(gca, 'FontSize', 13)
for i=1:length(dt)
    plot(x{i}, gain{i}, '-o', 'LineWidth', 1)
    hold on;
    xlabel('Time (s)')
    ylabel('Firing rate (spikes/s)')
end
title('Gains')
legend(['dt = ' num2str(dt(1) * 1000) ' ms'], ['dt = ' num2str(dt(2) * 1000) ' ms'], ['dt = ' num2str(dt(3) * 1000) ' ms']);

% plot modulated neural response
figure(3); 
set(gca, 'FontSize', 13)
for i=1:length(dt)
    plot(x{i}, lambda{i}, '-o', 'LineWidth', 1)
    hold on;
    xlabel('Time (s)')
    ylabel('Firing rate (spikes/s)')
end
title('Modulated neural response')
legend(['dt = ' num2str(dt(1) * 1000) ' ms'], ['dt = ' num2str(dt(2) * 1000) ' ms'], ['dt = ' num2str(dt(3) * 1000) ' ms']);
