% Simulate spike trains arising from a simple modulated Poisson process.

%% Setup
clear all
clc

%% Parameters
T = 5;            % duration (seconds)
dt = 1 / 1000;    % time bin (seconds)
rate = 40;        % mean rate (spikes/s)
var_gain = 0.1;   % variance of the gain
n_trials = 100;   % number of trials

% generate modulated Poisson process
x = unifrnd(0, 1, [n_trials, T / dt]);
r = 1 / var_gain;
s = var_gain;
G = repmat(gamrnd(r, s, [n_trials, 1]), 1, T / dt);
spike_train = double(x <= rate * G * dt);
% spike_train = double(x <= rate * dt);


% compute means and variances across trials
mean_trial = sum(spike_train, 2);
var_trial = var(spike_train, 0, 2) * (T / dt);

% plots    
subplot(1, 3, 1)
imagesc(spike_train); 
colormap('gray');
colorbar;
axis square; 
xlabel('Time (ms)')
ylabel('Trial number')
set(gca,'YDir','normal')

subplot(1, 3, 2)
histogram(mean_trial, 'Normalization', 'percentage')
xlabel('Mean firing rate (spikes/s)')
ylabel('Probability (%)')
axis square; 

subplot(1, 3, 3)
scatter(mean_trial, var_trial)
xlabel('Mean (spikes/s')
ylabel('Variance (spikes/s)^2')
set(gca,'YDir','normal', 'XScale', 'log', 'YScale', 'log')
xlim([min([mean_trial, var_trial], [], 'all') max([mean_trial, var_trial], [], 'all')])
ylim([min([mean_trial, var_trial], [], 'all') max([mean_trial, var_trial], [], 'all')])
axis square; 

%% Compute across-trial autocorrelation
across_trial_spikes = sum(spike_train, 1); % accross trial spike count
mean_spike_count = mean(spike_train, 1);
var_spike_count = var(spike_train, 0, 1);

R = xcorr(across_trial_spikes - mean_spike_count);
figure; plot(R)