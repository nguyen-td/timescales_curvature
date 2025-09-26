% Test how the Fano factor changes as a function of bin size

%% Setup
clear all
clc

seed = 50;
rng(seed)

%% Define random processes
n_frames = 11;           % number of video frames
frame_duration = 0.2;    % duration over with a single frame was shown (seconds)
n_trials = 1000;         % number of trials

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = 0.001;                       % simulation time steps (in s)
x = linspace(0, T, T / dt);       % dummy data points

% parameters of fast population
n_neurons = 10;    % number of neurons within the population
rho_f = 2;         % variance of tuning curves
tau_f = 0.2;       % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = 0.1;  % [min, max] variance of gain
tau_g = 0.2;  % time scale of gain (seconds)
q_g = 2;      % power law exponent of gain covariance function

%% Simulate neural responses
[tuning_curves, gain, lambda, K_g] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f, rho_g, tau_g, q_g);

%% Generate spike trains
spike_probs = unifrnd(0, 1, size(tuning_curves));
spikes_CMP = double(spike_probs <= lambda * dt);

%% Compute mean and variance over time bins

% find suitable bin sizes
clear bin_sizes_all
count = 1;
for bin = 0:dt:T
    if mod(size(tuning_curves, 2), bin / dt) == 0
        bin_sizes_all(count) = bin;
        count = count + 1;
    end
end

mean_counts_bin = {}; % means within bins
for ibin = 1:numel(bin_sizes_all)
    bin_factor = bin_sizes_all(ibin) / dt;
    
    % downsampling
    spikes_reshaped = reshape(spikes_CMP, size(spikes_CMP, 1), bin_factor, [], n_trials);  % n_neurons x bin_size x n_bins x n_trials
    binned_spikes = sum(spikes_reshaped, 2);                  % n_neurons x 1 x n_bins x n_trials
    
    % compute mean
    mean_counts_bin{ibin} = squeeze(mean(binned_spikes, 4));  % mean spikes per bin across trials

    % compute variance
    var_counts_bin{ibin} = squeeze(var(binned_spikes, 0, 4)); % variance per bin across trials

    % compute Fano factor
    fano_counts_bin{ibin} = var_counts_bin{ibin} ./ mean_counts_bin{ibin};
end

% compute Fano factor averaged over time bins and neurons for each bin size
fano_counts = zeros(n_neurons, numel(bin_sizes_all));
for ibin = 1:numel(bin_sizes_all)
    fano_counts(:, ibin) = mean(fano_counts_bin{ibin}, 2, 'omitnan');
end


%% Compute means and variances from analytical expressions

mean_GP = {};
var_GP = {};
% kappa = {};
for k = 1:length(bin_sizes_all)
    [mean_GP{k}, var_GP{k}] = mean_var_CMP(tuning_curves, rho_g, dt, T, bin_sizes_all(k), tau_g, q_g);
end

% compute Fano factor averaged over time bins and neurons for each bin size
fano_CMP = zeros(n_neurons, numel(bin_sizes_all));
for ibin = 1:numel(bin_sizes_all)
    % fano_CMP(:, ibin) = 1 + mean(var_GP{ibin}, 2, 'omitnan') .* (kappa{ibin} - 1);
    fano_CMP(:, ibin) = mean(var_GP{ibin}, 2, 'omitnan') ./ mean(mean_GP{ibin}, 2, 'omitnan');
end

%% Plot
itrial = 1;
n_plots = 5; % number of neurons to plot side by side

figure(1)
tiledlayout(4, n_plots, 'TileSpacing', 'compact', 'Padding', 'none');
for il = 1:n_plots
    % subplot(2, n_plots, il)
    nexttile;
    plot(x, squeeze(lambda(il, :, itrial)), 'LineWidth', 1)
    ylim([min(lambda, [], 'all') max(lambda, [], "all")])
    ylabel('\lambda(t) (spikes/second)')
    xlabel('Time (s)')
    axis square;   
end
lgd_lambda = legend('\lambda(t)', 'Location', 'northwest', 'IconColumnWidth', 10);
lgd_lambda.IconColumnWidth = 15;
lgd_lambda.Box = 'off';

for is = 1:n_plots
    % subplot(2, n_plots, is + n_plots)
    nexttile;
    imagesc(squeeze(spikes_CMP(is, :, :))'); 
    colormap('gray');
    colorbar;
    axis square; 
    xlabel('Time (ms)')
    ylabel('Trial number')
    set(gca,'YDir','normal')
end

for im = 1:n_plots
    nexttile
    scatter(bin_sizes_all, fano_counts(im, :))
    hold on
    ylabel('Fano Factor')
    xlabel('Time bin size (s)')
    axis square;
    title('Empirical Fano factor')
    ylim([min(fano_counts, [], 'all') max(fano_counts, [], "all")])
end

for im = 1:n_plots
    nexttile
    scatter(bin_sizes_all, fano_CMP(im, :))
    ylabel('Fano Factor')
    xlabel('Time bin size (s)')
    axis square;
    title('Analytical Fano factor')
    ylim([min(fano_counts, [], 'all') max(fano_counts, [], "all")])
end

%% Plot empirical vs. analytical solutions
figure(2)
tiledlayout(3, 2);

% plot of the mean
min_mean = min([cell2mat(mean_counts_bin), cell2mat(mean_GP)], [], 'all');
max_mean = max([cell2mat(mean_counts_bin), cell2mat(mean_GP)], [], 'all');

nexttile
for ibin = 1:length(mean_counts_bin)
    plot(linspace(0, max_mean, numel(mean_counts_bin{ibin}(im, :))), linspace(0, max_mean, numel(mean_counts_bin{ibin}(im, :))), 'k') 
    hold on;
    scatter(mean_counts_bin{ibin}(im, :), mean_GP{ibin}(im, :))
    xlabel('Empirical mean (spikes)')
    ylabel('Analytical mean (spikes)')
    axis square;
    title('Mean')
    xlim([min_mean, max_mean])
    ylim([min_mean, max_mean])
end

% plot of the variance
min_var = min([cell2mat(var_counts_bin), cell2mat(var_GP)], [], 'all');
max_var = max([cell2mat(var_counts_bin), cell2mat(var_GP)], [], 'all');

nexttile
for ibin = 1:length(mean_counts_bin)
    plot(linspace(0, max_var, numel(var_counts_bin{ibin}(im, :))), linspace(0, max_var, numel(var_counts_bin{ibin}(im, :))), 'k') 
    hold on;
    scatter(var_counts_bin{ibin}(im, :), var_GP{ibin}(im, :))
    xlabel('Empirical variance (spikes^{2})')
    ylabel('Analytical variance (spikes^{2})')
    axis square;
    title('Variance')
    xlim([min_var, max_var])
    ylim([min_var, max_var])
end

% empirical Fano factor
nexttile
scatter(bin_sizes_all, mean(fano_counts, 1))
title('Empirical Fano factor (neuron average)')
ylabel('Fano Factor')
xlabel('Time bin size (s)')
axis square;

% analytical Fano factor
nexttile
scatter(bin_sizes_all, mean(fano_CMP, 1))
title('Analytical Fano factor (neuron average)')
ylabel('Fano Factor')
xlabel('Time bin size (s)')
axis square;

% compare empirical with analytical Fano factor
fano_counts_sorted = sort(mean(fano_counts, 1));
fano_CMP_sorted = sort(mean(fano_CMP, 1));

nexttile
plot(bin_sizes_all, fano_counts_sorted, 'LineWidth', 1.5) 
hold on;
plot(bin_sizes_all, fano_CMP_sorted, 'LineWidth', 1.5) 
title('Empirical Fano factor (neuron average)')
ylabel('Fano Factor')
xlabel('Time bin size (s)')
axis square;

legend('Empirical', 'Analytical')

nexttile
scatter(bin_sizes_all, mean(fano_counts, 1)) 
hold on;
scatter(bin_sizes_all, mean(fano_CMP, 1)) 
title('Empirical Fano factor (neuron average)')
ylabel('Fano Factor')
xlabel('Time bin size (s)')
axis square;

legend('Empirical', 'Analytical')