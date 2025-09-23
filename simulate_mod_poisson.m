% Function to simulate neural responses from a modulated Poisson process
% where tuning functions and gains are modeled as Gaussian processes.
%
% Inputs:
%   n_trials  - [int] Number of trials
%   x         - (1 x T / dt) Time bins
%   n_neurons - [int] Number of neurons
%   rho_f     - [double] Variance of covariance function of tuning curve
%   tau_f     - [double] Length scale of covariance function of tuning curve
%   rho_g     - [double] Variance of covariance function of gain
%   tau_g     - [double] Length scale of covariance function of gain
%   q_g       - [double] Power law exponent of covariance function of gain
%
% Outputs:
%   tuning_curves - (n_neurons x length(x) x n_trials) Array of simulated tuning curves
%   gain          - (n_neurons x length(x) x n_trials) Array of simulated gains
%   lambda        - (n_neurons x length(x) x n_trials) Array of average firing rates
%   K_g           - (length(x) x length(x)) Covariance matrix of gain realizations 

function [tuning_curves, gain, lambda, K_g] = simulate_mod_poisson(n_trials, x, n_neurons, rho_f, tau_f, rho_g, tau_g, q_g)    

    % define mean vector and covariance function for gain
    % kernel_g = @(t1, t2, rho, tau, q) rho * exp(-1/2 * abs((t1 - t2) / tau).^q); % exponentiated power law (EPL) kernel
    K_g = epl_kernel(x, x', rho_g, tau_g, q_g);
    % autocorr_g = @(delta, tau, q) exp(-1/2 * abs(delta / tau).^q);
    
    % define mean vector and covariance function for tuning curve
    % kernel_f = @(t1, t2, rho, tau) rho * exp(-(t1 - t2).^2 / (2 * tau^2)); % RBF kernel
    K_f = rbf_kernel(x, x', rho_f, tau_f);

    % one tuning curve per neuron within the population, do not change over trials
    mu_f = zeros(1, length(x));
    tuning_curves = repmat(exp(mvnrnd(mu_f, K_f, n_neurons)), 1, 1, n_trials); % n_neurons x length(x) x n_trials 
    
    % gain varies over trials and neurons
    mu_g = zeros(1, length(x));
    gain = permute(reshape(exp(mvnrnd(mu_g, K_g, n_neurons * n_trials)), [n_neurons, n_trials, length(x)]), [1, 3, 2]); % n_neurons x length(x) x n_trials 
    
    % compute modulated response
    lambda = tuning_curves .* gain;
end