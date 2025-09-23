%% Setup
clear all
clc

%% Simulate hierarchy of neural populations

n_frames = 11;          % number of video frames
frame_duration = 0.2;   % duration over with a single frame was shown (seconds)
n_trials = 100;         % number of trials

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = 0.001;                       % bin size (in s)
x = linspace(0, T, T / dt);       % dummy data points

% parameters of fast population
n_neurons = 10;      % number of neurons within the population
rho_f = 0.2;         % variance of tuning curves
tau_f = 1.0;         % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = 0.1;  % [min, max] variance of gain
tau_g = [0.002, 0.02, 0.2, 2]; % time scale of gain (seconds)
q_g = 2;      % power law exponent of gain covariance function

% loop over hierarchy of neural populations
c = zeros(1, length(tau_g));
for ip = length(tau_g)
    [tuning_curves, gain, lambda, K_g] = simulate_mod_poisson(n_trials, x, n_neurons, rho_f, tau_f, rho_g, tau_g(ip), q_g);
    
    % compute curvature
    mean_process = tuning_curves .* exp(rho_g/2);
    var_process = mean_process + mean_process.^2 .* (1 / length(x)^2 * sum(sum(exp(K_g))) - 1);

    % compute embedding on trial averages
    y = 2 ./ mean(var_process, 3) .* asinh(mean(var_process, 3) .* sqrt(mean(lambda, 3)));
    c(ip) = mean(rad2deg(compute_trajectory_pixel(reshape(y, [1, size(y, 1), size(y, 2)])))); % curvature in degrees
end