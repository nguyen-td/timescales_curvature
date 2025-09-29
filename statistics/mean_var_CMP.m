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
%   mean_lambda   - (n_neurons x n_bins) Mean spike count per time bin
%   var_lambda    - (n_neurons x n_bins) Variance per time bin
%   mean_gain     - [double] Mean of gain
%   var_gain      - (1 x n_bins) Variance of gain per time bin

function [mean_lambda, var_lambda, mean_gain, var_gain] = mean_var_CMP(tuning_curves, rho_g, dt, T, bin_size, tau_g, q_g)
    
    % tuning curves are equal across trials
    tuning_curves = squeeze(tuning_curves(:, :, 1)); 

     % downsampling
    bin_factor = bin_size / dt;
    tuning_curves_reshaped = reshape(tuning_curves, size(tuning_curves, 1), bin_factor, []);  % (n_neurons x bin_factor x n_bins)
    
    % compute mean of lambda
    mean_lambda = squeeze(exp(rho_g/2) .* sum(tuning_curves_reshaped, 2)) * dt;
    
    % get variance of lambda and gain
    [integral_lambda, var_gain] = compute_integrals(tuning_curves, tuning_curves_reshaped, bin_factor, T, tau_g, rho_g, q_g, dt, bin_size); 
    var_lambda = mean_lambda + integral_lambda;

    % compute mean of gain
    mean_gain = exp(1/2 * rho_g); % K_g is a stationary kernel that only depends on time lag (t1-t2) so doesn't depend on bin size; if time lag = 0, covariance collapses to rho_g
    
end

function [gamma, var_gain] = compute_integrals(tuning_curves, tuning_curves_reshaped, bin_factor, T, tau_g, rho_g, q_g, dt, dt_bin)

    rng(42)
    
    time_total = linspace(0, T, size(tuning_curves, 2));
    time_bins = reshape(time_total, bin_factor, []);
    [n_neurons, bin_size, ~] = size(tuning_curves_reshaped);

    gamma = zeros(n_neurons, size(time_bins, 2));
    var_gain = zeros(1, size(time_bins, 2));
    % mean_gain = zeros(1, size(time_bins, 2));
    for ibin = 1:size(gamma, 2)

        % generate grid points
        [T1, T2] = meshgrid(time_bins(:, ibin), time_bins(:, ibin));

        K_g = exp(epl_kernel(T1, T2, rho_g, tau_g, q_g));

        % compute gamma for variance of lambda
        tuning_squared = reshape(tuning_curves_reshaped(:, :, ibin), [n_neurons, bin_size, 1]) .* reshape(tuning_curves_reshaped(:, :, ibin), [n_neurons, 1, bin_size]);
        gamma(:, ibin) = exp(rho_g) * dt^2 * sum(tuning_squared .* (permute(repmat(K_g, 1, 1, n_neurons), [3, 2, 1]) - 1), [2, 3]); 
        
        % % compute mean of gain
        % mean_gain(ibin) = sum(exp(1/2 * rho_g));

        % compute variance of gain
        var_gain(ibin) = exp(rho_g) * (dt^2 / dt_bin^2) * (sum(K_g - 1, 'all'));
    end
end
