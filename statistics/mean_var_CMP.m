% Compute mean and variance of spike counts under the CMP model.
%
% Inputs:
%   tuning_curves - (n_neurons x length(x) x n_trials) Array of simulated tuning curves
%   K_g           - (length(x) x length(x)) Covariance matrix of gain realizations 
%   rho_g         - [double] Variance of covariance function of gain
%   T             - [double] Duration (in seconds)
%   dt            - [double] Sampling factor (in seconds)
%   bin_size      - [double] Bin size (in seconds)
%   tau_g         - [double] Length scale of the gain
%   rho_g         - [double] Variance of gain
%   q             - [double] (0 < q <= 2) Exponent

function [mean_GP, var_GP, kappa] = mean_var_CMP(tuning_curves, rho_g, bin_size, tau_g, q_g)

    tuning_curves = squeeze(tuning_curves(:, :, 1)); % tuning curves are equal across trials
    
    % Compute mean
    mean_GP = sum(tuning_curves .* exp(rho_g/2) * bin_size, 2);

    % Compute variance
    kappa = compute_kappa(bin_size, tau_g, rho_g, q_g);
    var_GP = mean_GP + mean_GP.^2 * (kappa - 1);

end

function kappa = compute_kappa(bin_size, tau_g, rho_g, q_g)

    % grid points
    h = bin_size / 10000; % sampling interval to discretize integral
    t = 0:h:bin_size;

    [T1, T2] = meshgrid(t, t);
    
    vals = exp(epl_kernel(T1, T2, rho_g, tau_g, q_g));
    kappa = (h^2 / bin_size^2) * sum(vals(:));
end