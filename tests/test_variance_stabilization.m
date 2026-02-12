%% Setup
clear all
clc

seed = 67;
rng(seed)

%% Draw RVs from different types of Poisson distributions and see mean-variance relationship
n_trials = 1000;
lambda = 0:200;

lambda_mat = repmat(lambda.', 1, n_trials);

% Poisson and square-root transformed Poisson variable
poisson_rv = poissrnd(lambda_mat);

mu_poisson_rv = mean(poisson_rv, 2);
var_poisson_rv = var(poisson_rv, 0, 2);
mu_poisson_rv_sqrt = mean(sqrt(poisson_rv));
var_poisson_rv_sqrt = var(sqrt(poisson_rv));

figure(1); 
plot(mu_poisson_rv, var_poisson_rv)
hold on;
plot(mu_poisson_rv_sqrt, var_poisson_rv_sqrt)
ylim([0 max(sqrt(poisson_rv), [], "all")])

% Classic modulated Poisson
figure(3);
var_gain = [0.01, 0.1, 1.0];   % variance of the gain
for iG = 1:length(var_gain)
    r = 1 / var_gain(iG);
    s = var_gain(iG);
    G = repmat(gamrnd(r, s, [n_trials, 1]), 1, length(lambda))';

    mod_poisson_rv = poissrnd(lambda_mat .* G);
    plot(mean(mod_poisson_rv, 2), var(mod_poisson_rv, 0, 2))
    hold on;
end

% Classic modulated Poisson after transformation
figure(4)
var_gain = [0.01, 0.1, 1.0];   % variance of the gain
for iG = 1:length(var_gain)
    sigma = diag(repmat(var_gain(iG), 1, length(lambda)));
    mu = -0.5 * diag(sigma);
    gain = exp(mvnrnd(mu, sigma, n_trials))';

    mod_poisson_rv = poissrnd(lambda_mat .* gain);
    % mod_poisson_hat_rv = 2 / exp(diag(sigma) - 1) * asinh(exp(diag(sigma) - 1) .* sqrt(mod_poisson_rv));
    plot(mean(mod_poisson_rv, 2), var(mod_poisson_rv, 0, 2))
    hold on;
end