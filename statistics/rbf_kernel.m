% Radial basis function (RBF) kernel.
%
% Inputs: 
%   t1  - (1 x T) Time series
%   t2  - (1 x T) Time series
%   rho - [double] Variance
%   tau - [double] Length scale
%
% Output:
%   K - (T x T) Covariance matrix

function K = rbf_kernel(t1, t2, rho, tau)
    K = rho * exp(-(t1 - t2).^2 / (2 * tau^2)); % RBF kernel
end