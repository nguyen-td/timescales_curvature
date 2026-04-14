%% Setup
clear all
clc

seed = 3;
rng(seed)

%% Simulate hierarchy of neural populations

n_frames = 11;        % number of video frames
frame_duration = 0.2; % duration over with a single frame was shown (seconds)
n_trials = 100;      % number of trials
is_analytical = true; % estimate curvature based on analytical solutions or simulated spike counts

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = 0.001;                       % simulation time steps (in s)
x = linspace(0, T, T / dt);       % dummy data points

% parameters of fast population
n_neurons = 100;                  % number of neurons within the population
rho_f = 3;                       % variance of tuning curves
tau_f = linspace(0.001, 2, 100);  % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = 0.1;           % [min, max] variance of gain
tau_g = [0.002, 0.02]; % time scale of gain (seconds)
q_g = 2;               % power law exponent of gain covariance function
% tau_g = 0.02;          % time scale of gain (seconds)
% q_g = [0.5, 2];        % power law exponent of gain covariance function

%% Simulate neural responses

% find suitable bin sizes
clear bin_sizes_all
count = 1;
for bin = 0:dt:T
    if mod(numel(x), bin / dt) == 0
        bin_sizes_all(count) = bin;
        count = count + 1;
    end
end

% loop over hierarchy of neural populations
c_slow_gain = zeros(numel(bin_sizes_all), numel(tau_f));
c_fast_gain = zeros(numel(bin_sizes_all), numel(tau_f));

for itau = 1:numel(tau_f)
    [tuning_curves_fast_gain, ~, lambda_fast_gain] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g(1), q_g);
    [tuning_curves_slow_gain, ~, lambda_slow_gain] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g(2), q_g);
    % [tuning_curves_fast_gain, ~, ~] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g, q_g(1));
    % [tuning_curves_slow_gain, ~, ~] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g, q_g(2));

    % loop over different bin sizes
    for ibin = 1:numel(bin_sizes_all)
        if is_analytical
            [mean_lambda_fast_gain, ~, ~, var_gain_fast_gain] = mean_var_CMP(tuning_curves_fast_gain, rho_g, dt, T, bin_sizes_all(ibin), tau_g(1), q_g);
            [mean_lambda_slow_gain, ~, ~, var_gain_slow_gain] = mean_var_CMP(tuning_curves_slow_gain, rho_g, dt, T, bin_sizes_all(ibin), tau_g(2), q_g);
            % [mean_lambda_fast_gain, ~, ~, var_gain_fast_gain] = mean_var_CMP(tuning_curves_fast_gain, rho_g, dt, T, bin_sizes_all(ibin), tau_g, q_g(1));
            % [mean_lambda_slow_gain, ~, ~, var_gain_slow_gain] = mean_var_CMP(tuning_curves_slow_gain, rho_g, dt, T, bin_sizes_all(ibin), tau_g, q_g(2));
    
            % compute global curvature (in degrees)
            y_fast_gain = compute_VST(var_gain_fast_gain, mean_lambda_fast_gain);
            y_slow_gain = compute_VST(var_gain_slow_gain, mean_lambda_slow_gain);
            c_fast_gain(ibin, itau) = rad2deg(mean(compute_curvature(y_fast_gain)));
            c_slow_gain(ibin, itau) = rad2deg(mean(compute_curvature(y_slow_gain)));
        else
            % generate spikes
            spike_probs_fast_gain = unifrnd(0, 1, size(lambda_fast_gain));
            spikes_fast_gain = double(spike_probs_fast_gain <= lambda_fast_gain * dt);
            bin_factor_fast_gain = bin_sizes_all(ibin) / dt;

            spike_probs_slow_gain = unifrnd(0, 1, size(lambda_slow_gain));
            spikes_slow_gain = double(spike_probs_slow_gain <= lambda_slow_gain * dt);
            bin_factor_slow_gain = bin_sizes_all(ibin) / dt;
            
            % downsampling
            spikes_reshaped_fast_gain = reshape(spikes_fast_gain, size(spikes_fast_gain, 1), bin_factor_fast_gain, [], n_trials);  % n_neurons x bin_size x n_bins x n_trials
            binned_spikes_fast_gain = sum(spikes_reshaped_fast_gain, 2); % n_neurons x 1 x n_bins x n_trials (spike counts per bin)
            
            spikes_reshaped_slow_gain = reshape(spikes_slow_gain, size(spikes_slow_gain, 1), bin_factor_slow_gain, [], n_trials);  % n_neurons x bin_size x n_bins x n_trials
            binned_spikes_slow_gain = sum(spikes_reshaped_slow_gain, 2); % n_neurons x 1 x n_bins x n_trials (spike counts per bin)

            % compute spike count variance
            var_spikes_fast_gain = squeeze(var(binned_spikes_fast_gain, 0, 4)); % variance per bin across trials
            var_spikes_slow_gain = squeeze(var(binned_spikes_slow_gain, 0, 4)); % variance per bin across trials
            
            % normalized spike counts (variance = 1)
            norm_spikes_fast_gain = squeeze(binned_spikes_fast_gain) ./ sqrt(var_spikes_fast_gain);
            mean_norm_spikes_fast_gain = mean(norm_spikes_fast_gain, 3);
            % var_norm_spikes_fast_gain = var(norm_spikes_fast_gain, 0, 3);

            norm_spikes_slow_gain = squeeze(binned_spikes_slow_gain) ./ sqrt(var_spikes_slow_gain);
            mean_norm_spikes_slow_gain = mean(norm_spikes_slow_gain, 3);
            % var_norm_spikes_slow_gain = var(norm_spikes_slow_gain, 0, 3);

            % compute curvature
            c_fast_gain(ibin, itau) = rad2deg(mean(compute_curvature(mean_norm_spikes_fast_gain)));
            c_slow_gain(ibin, itau) = rad2deg(mean(compute_curvature(mean_norm_spikes_slow_gain)));
        end
    end
end

%% Plot results

n_rows = 6;
n_cols = floor(numel(bin_sizes_all) / n_rows);
max_curvature = max([c_fast_gain, c_slow_gain], [], 'all');

figure(1)
tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'none');
set(gca, 'FontSize', 13)

for ibin = 1:numel(bin_sizes_all)
    nexttile
    scatter(tau_f, c_slow_gain(ibin, :), 10, 'filled')
    hold on;
    scatter(tau_f, c_fast_gain(ibin, :), 10, 'filled')
    title(['\Deltat = ' num2str(bin_sizes_all(ibin)) 's'])
    xlabel('Time scale of f(s,t) (s)')
    ylabel('Curvature (deg)')

    legend('Slow gain', 'Fast gain')
end