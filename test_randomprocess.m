%% Setup
clear all
clc

%% Define random processes
n_frames = 11;           % number of video frames
frame_duration = 0.2;    % duration over with a single frame was shown (seconds)
n_trials = 100;         % number of trials

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = 0.001;                       % bin size (in s)
x = linspace(0, T, T / dt);       % dummy data points

% parameters of fast population
n_neurons = 10;      % number of neurons within the population
rho_f = 2;           % variance of tuning curves
tau_f = 1.0;         % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = 3;      % [min, max] variance of gain
tau_g = 0.02;   % time scale of gain (seconds)
q_g = 2;        % power law exponent of gain covariance function

% define mean vector and covariance function for gain
kernel_g = @(t1, t2, rho, tau, q) rho * exp(-1/2 * abs((t1 - t2) / tau).^q); % exponentiated power law (EPL) kernel
K_g = kernel_g(x, x', rho_g, tau_g, q_g);
autocorr_g = @(delta, tau, q) exp(-1/2 * abs(delta / tau).^q);

% define mean vector and covariance function for tuning curve
kernel_f = @(t1, t2, rho, tau) rho * exp(-(t1 - t2).^2 / (2 * tau^2)); % RBF kernel
K_f = kernel_f(x, x', rho_f, tau_f);

%% Simulate neural responses
% one tuning curve per neuron within the population, do not change over trials
mu_f = zeros(1, length(x));
tuning_curves = repmat(mvnrnd(mu_f, K_f, n_neurons), 1, 1, n_trials); % n_neurons x length(x) x n_trials 

% gain varies over trials, shared across neurons
mu_g = zeros(1, length(x));
gain = permute(repmat(mvnrnd(mu_g, K_g, n_trials), 1, 1, n_neurons), [3, 2, 1]); % n_neurons x length(x) x n_trials 



% compute gain and autocorrelation
delta = linspace(-T, T, 2 * length(x) + 1);
figure; plot(autocorr_g(delta, tau_g, q_g))
figure; plot(x, gain)
