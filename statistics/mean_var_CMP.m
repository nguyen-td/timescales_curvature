% Compute mean and variance of spike counts under the CMP model.
%
% Inputs:
%   tuning_curves - (n_neurons x length(x) x n_trials) Array of simulated tuning curves
%   K_g           - (length(x) x length(x)) Covariance matrix of gain realizations 
%   rho_g         - [double] Variance of covariance function of gain
%   dt            - [double] Simulation time steps (in s)
%   bin_size      - [double] Bin size (in seconds)
%   tau_g         - [double] Length scale of the gain
%   rho_g         - [double] Variance of gain
%   q             - [double] (0 < q <= 2) Exponent
%
% Outputs:
%   mean_GP       - (n_neurons x length(x) / bin_factor) Mean spike countper time bin
%   var_GP        - (n_neurons x length(x) / bin_factor) Variance per time bin
%   kappa         - [double] Fluctuation factor

function [mean_GP, var_GP, kappa] = mean_var_CMP(tuning_curves, rho_g, dt, bin_size, tau_g, q_g)
    
    % Tuning curves are equal across trials
    tuning_curves = squeeze(tuning_curves(:, :, 1)); 

     % Downsampling
    bin_factor = bin_size / dt;
    signal_reshaped = reshape(tuning_curves, size(tuning_curves, 1), bin_factor, []);  % (n_neurons x bin_factor x length(x)/bin_factor)
    tuning_curves_new = squeeze(mean(signal_reshaped, 2));
    
    % Compute mean
    mean_GP = tuning_curves_new .* exp(rho_g/2) * bin_size;

    % Compute variance
    kappa = compute_kappa(bin_size, tau_g, rho_g, q_g);
    var_GP = mean_GP + mean_GP.^2 .* (kappa - 1);

end

function kappa = compute_kappa(bin_size, tau_g, rho_g, q_g)

    % grid points
    h = bin_size / 10000; % sampling interval to discretize integral
    t = 0:h:bin_size;

    [T1, T2] = meshgrid(t, t);
    
    vals = exp(epl_kernel(T1, T2, rho_g, tau_g, q_g));
    kappa = (h^2 / bin_size^2) * sum(vals(:));
end