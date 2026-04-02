% Simulate spike trains arising from a simple modulated Poisson process.

%% Setup
clear all
clc

%% Parameters
T = 2;            % duration (seconds)
dt = 1 / 1000;    % time bin (seconds)
% rate = 70;        % mean rate (spikes/s)
rate = linspace(1, 100, T / dt);
var_gain = 0.1;   % variance of the gain
n_trials = 1000;   % number of trials
bin_size = 0.1;   % time bin in seconds
gain_distribution = 'gamma'; % either 'gamma' or 'lognormal'

% generate modulated Poisson process
if strcmpi(gain_distribution, 'lognormal')
    Sigma_G = diag(var_gain);
    mu_G = -1/2 * diag(Sigma_G);
    G = repmat(exp(normrnd(mu_G, Sigma_G, [n_trials, 1])), 1, T / dt);
else
    r = 1 / var_gain;
    s = var_gain;
    G = repmat(gamrnd(r, s, [n_trials, 1]), 1, T / dt);
end

x = unifrnd(0, 1, [n_trials, T / dt]);
% lambda = rate;
lambda = rate .* G;
% lambda = 2 / (exp(diag(Sigma_G)) - 1) * asinh((exp(diag(Sigma_G)) - 1) * sqrt(rate .* G));
spike_train = double(x <= lambda * dt);

% compute means and variances across trials
[mean_spike_count, var_spike_count] = get_mean_var(spike_train, 100);
max_value = max([mean_spike_count, var_spike_count], [], 'all');

%% Plotting   
subplot(1, 2, 1)
imagesc(spike_train); 
colormap('gray');
colorbar;
axis square; 
xlabel('Time (ms)')
ylabel('Trial number')
set(gca,'YDir','normal')

subplot(1, 2, 2)
plot([0, max_value], [0, max_value], 'k--')
hold on;
scatter(mean_spike_count, var_spike_count)
xlabel('Mean spike count')
ylabel('Variance of spike count')
set(gca,'YDir','normal', 'XScale', 'log', 'YScale', 'log')
% xlim([min([mean_spike_count, var_spike_count], [], 'all') max([mean_spike_count, var_spike_count], [], 'all')])
% ylim([min([mean_spike_count, var_spike_count], [], 'all') max([mean_spike_count, var_spike_count], [], 'all')])
axis square; 
grid on

function [mean_win, var_win] = get_mean_var(spike_train, n_windows)
    mean_win = zeros(n_windows, 1);
    var_win = zeros(n_windows, 1);
    for iwin = 1:n_windows
        issucess = false;
        while ~issucess
            start_win = randi(size(spike_train, 2));
            size_win = randi(size(spike_train, 2));
            if (start_win + size_win) <= size(spike_train, 2) % if the window has a plausible size (does not exceed the time axis)
                issucess = true;
            end
        end
        mean_win(iwin) = mean(sum(spike_train(:, start_win:start_win+size_win), 2));
        var_win(iwin) = var(sum(spike_train(:, start_win:start_win+size_win), 2));
    end
end