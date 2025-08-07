%% Setup
clear all
clc

%% Define random processes
T = 1;                      % duration (seconds)
dt = 1 / 1000;              % time bin
x = linspace(0, T, T / dt); % dummy data points
rho_g = 0.1;                % variance of gain
tau_g = 0.02;               % time scale of gain (seconds)
q_g = 2;                    % power law exponent of gain covariance function

mu_g = zeros(1, length(x));
rbf_g = @(t1, t2, rho, tau, q) rho * exp(-1/2 * abs((t1 - t2) / tau).^q);
K_g = rbf_g(x, x', rho_g, tau_g, q_g);
autocorr_g = @(delta, tau, q) exp(-1/2 * abs(delta / tau).^q);

% compute gain and autocorrelation
gain = mvnrnd(mu_g, K_g);
delta = linspace(-T, T, 2 * length(x) + 1);
plot(autocorr_g(delta, tau_g, q_g))
