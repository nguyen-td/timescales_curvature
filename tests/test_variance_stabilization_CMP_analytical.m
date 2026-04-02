%% Test variance-stabilizing transformation for CMP.

%% Setup
clear all
clc

seed = 3;
rng(seed)

%% Define random processes
n_frames = 11;           % number of video frames
frame_duration = 0.2;    % duration over with a single frame was shown (seconds)
n_trials = 1000;

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = 0.001;                       % simulation time steps (in s)
x = linspace(0, T, T / dt);       % dummy data points

% parameters of fast population
n_neurons = 100;    % number of neurons within the population
rho_f = 3;         % variance of tuning curves
tau_f = 0.2;       % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = 0.1;  % [min, max] variance of gain
tau_g = 0.2;  % time scale of gain (seconds)
q_g = 2;      % power law exponent of gain covariance function

%% Simulate neural responses
mean_frange = [5, 50]; % determine the range of the firing rate, can be used to force realistic ranges
[tuning_curves, gain, lambda, K_g] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f, rho_g, tau_g, q_g, mean_frange);

%% Find suitable bin sizes
clear bin_sizes_all
count = 1;
for bin = 0:dt:T
    if mod(size(tuning_curves, 2), bin / dt) == 0
        bin_sizes_all(count) = bin;
        count = count + 1;
    end
end

%% Compute means and variances from analytical expressions
for ibin = 1:numel(bin_sizes_all)
    [mean_spike_count{ibin}, var_spike_count{ibin}, mean_gain{ibin}, var_gain{ibin}] = mean_var_CMP(tuning_curves, rho_g, dt, T, bin_sizes_all(ibin), tau_g, q_g);
end
y = compute_VST(var_gain{1}, lambda);

%% Generate spike trains from transformed rate
bin_size_idx = 12;
n_trials = 1000;
spike_probs = unifrnd(0, 1, [n_neurons, size(y, 2), n_trials]);
y_spikes = double(spike_probs <= y * dt);
bin_factor = bin_sizes_all(bin_size_idx) / dt;

% downsampling
spikes_reshaped = reshape(y_spikes, size(y_spikes, 1), bin_factor, [], n_trials);  % n_neurons x bin_size x n_bins x n_trials
binned_spikes = sum(spikes_reshaped, 2); % n_neurons x 1 x n_bins x n_trials

% compute mean and variance
mean_y_spikes = squeeze(mean(binned_spikes, 4));  % mean spikes per bin across trials
var_y_spikes = squeeze(var(binned_spikes, 0, 4)); % variance per bin across trials

%% Plot spike trains
n_plots = 10; % number of neurons to plot side by side
figure(1)
for is = 1:n_plots
    if is < 6
        nexttile;
        imagesc(squeeze(y_spikes(is, :, :))'); 
        colormap('gray');
        colorbar;
        axis square; 
        xlabel('Time (ms)')
        ylabel('Trial number')
        set(gca,'YDir','normal')
        title(['Neuron ' num2str(is)])
    else
        nexttile
        plot(squeeze(y(is-5, :, :)))
        axis square
        xlabel('Time (ms)')
        ylabel('Firing rate (spikes/s)')
        set(gca,'YDir','normal')
    end
end

%% Plot mean-variance relationship of the transformed firing rate
max_value = max([mean_spike_count{bin_size_idx}, var_y_spikes], [], 'all');
% max_value = max([mean_y_spikes, var_y_spikes], [], 'all');

figure(2)
scatter(mean_spike_count{bin_size_idx}, var_y_spikes)
% scatter(mean_y_spikes, var_y_spikes)
hold on;
plot([0, max_value], [0, max_value], 'k--')
xlabel('Mean (spikes)')
ylabel('Variance (spikes^{2})')
title(['Bin size: ' num2str(bin_sizes_all(bin_size_idx))])
axis square;
set(gca,'FontSize', 12)