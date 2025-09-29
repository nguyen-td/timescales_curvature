% Compute curvature betweeen rate vectors in discriminability space, based
% on Henaff et al. (2021).
%
% Inputs:
%   var_gain    - (n_neurons x n_bins) Variance of the gain
%   mean_lambda - (n_neurons x n_bins) Mean of the modulated rate function
%
% Outputs:
%   c - (1 x n_bins - 2) Curvature of neural embeddings

function c = compute_curvature(var_gain, mean_lambda)
    y = 2 ./ var_gain .* asinh(var_gain .* sqrt(mean_lambda)); % this embedding was derived for constant gain
    c = compute_trajectory_pixel(reshape(y, [1, size(y, 1), size(y, 2)]));

end