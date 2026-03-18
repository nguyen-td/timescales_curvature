% Compute curvature betweeen rate vectors in discriminability space, based
% on Henaff et al. (2021).
%
% Inputs:
%   var_gain    - (1 x n_bins) or (n_bins x n_trials) Variance of the gain
%   mean_lambda - (n_neurons x n_bins) Mean of the modulated rate function
%
% Outputs:
%   c - (1 x n_bins - 2)     Curvature of neural embeddings 
%   y - (n_neurons x n_bins) Embedding after variance-stabilizing transformation

function c = compute_curvature(y)
    c = compute_trajectory_pixel(reshape(y, [1, size(y, 1), size(y, 2)]));
end