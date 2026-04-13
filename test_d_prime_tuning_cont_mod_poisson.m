% Test discriminability from linear tuning curve for continuous modulated Poisson process.

%% Setup
clear
close all
clc

%% Define tuning curve
% define parameters
T = 2;                      % duration (seconds)
dt = 1 / 1000;              % time bin (seconds)
max_freq = 100;             % maximum firing rate
time_bins = [0.01, 0.02, 0.05, 0.1]; % time bins in seconds
min_rate = 20;
max_rate = 100; % 0-100 spk/s over T seconds

% parameters of fast population
rho_f = 3;     % variance of tuning curves
tau_f = 0.01;  % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = [0.01, 0.1];    % [min, max] variance of gain
tau_g = 0.002;   % time scale of gain (seconds)
q_g = 2;         % power law exponent of gain covariance function


%% Start simulation
orig_rate = linspace(min_rate, max_rate, T / dt); % create linear firing rate

figure(1)
colors = lines(numel(rho_g)); % distinct colors for each rho_g
t = tiledlayout(numel(time_bins), 3, 'TileSpacing', 'loose', 'Padding', 'compact');
for ibin = 1:numel(time_bins)

    time_bin_vec = linspace(0, T, T / time_bins(ibin));
    binned_rate = linspace(min_rate, max_rate, T / time_bins(ibin));
    mean_count = binned_rate .* time_bins(ibin);

    % reshape original rate to prepare for integration
    n_points_per_bin = time_bins(ibin) / dt;
    reshaped_rate = reshape(orig_rate, n_points_per_bin, []); % n_points_per_bin x n_bins

    % initialize d_prime and snr_transformed arrays
    d_prime = zeros(numel(rho_g), numel(binned_rate));
    snr_transformed = zeros(numel(rho_g), numel(binned_rate));
    
    % plot sdt_estimate and snr_transformed (estimates of d-prime)
    nexttile
    hold on, box off, axis square, grid on
    for irho_g = 1:numel(rho_g)
        % compute d_prime and snr_transformed
        [var_count, var_gain] = compute_integrals(mean_count, orig_rate, reshaped_rate, n_points_per_bin, T, tau_g, rho_g(irho_g), q_g, dt, time_bins(ibin));
        d_prime(irho_g, :) = (mean_count - mean_count(1)) ./ sqrt((var_count + var_count(1)) / 2);
    
        transformed_rate = 2 ./ var_gain .* asinh(var_gain .* sqrt(binned_rate .* time_bins(ibin)));
        snr_transformed(irho_g, :) = transformed_rate - transformed_rate(1);

        % plotting
        h(irho_g) = plot(time_bin_vec, d_prime(irho_g, :), 'Color', colors(irho_g, :), 'LineWidth', 1.4); 
        plot(time_bin_vec, snr_transformed(irho_g, :), '--', 'Color', colors(irho_g, :), 'LineWidth', 1.4)
    end
    legend(h, arrayfun(@(s) sprintf('\\rho_{g} = %.2f', s), rho_g, 'UniformOutput', false), 'Location', 'westoutside')
    title(['\Deltat = ' num2str(time_bins(ibin))])
    xlabel('Time (s)')
    ylabel("d'")

    % plot sdt_estimate / snr_transformed
    nexttile
    hold on, box off, axis square, grid on
    for irho_g = 1:numel(rho_g)
        plot(time_bin_vec, d_prime(irho_g, :) ./ snr_transformed(irho_g, :), 'Color', colors(irho_g,:), 'LineWidth', 1.4)
    end
    title('SDT estimate / SNR transformed', 'Units', 'normalized', 'Position', [0.5 1.03 0])
    xlabel('Time (s)')
    axis([0 2 0 2])

    % plot sdt_estimate - snr_transformed
    nexttile
    hold on, box off, axis square, grid on
    for irho_g = 1:numel(rho_g)
        plot(time_bin_vec, d_prime(irho_g, :) - snr_transformed(irho_g, :), 'Color', colors(irho_g,:), 'LineWidth', 1.4)
    end
    title('SDT estimate - SNR transformed', 'Units', 'normalized', 'Position', [0.5 1.03 0])
    xlabel('Time (s)')
    axis([0 T -2 2])
end
sgtitle('Solid = SDT (d''), Dashed = SNR (transformed)')

function [var_count, var_gain] = compute_integrals(mean_count, tuning_curves, tuning_curves_reshaped, bin_factor, T, tau_g, rho_g, q_g, dt, dt_bin)

    rng(42)
    
    time_total = linspace(0, T, size(tuning_curves, 2));
    time_bins = reshape(time_total, bin_factor, []);

    gamma = zeros(1, size(time_bins, 2));
    var_gain = zeros(1, size(time_bins, 2));
    % mean_gain = zeros(1, size(time_bins, 2));
    for ibin = 1:size(gamma, 2)

        % generate grid points
        [T1, T2] = meshgrid(time_bins(:, ibin), time_bins(:, ibin));

        K_g = exp(epl_kernel(T1, T2, rho_g, tau_g, q_g));

        % compute gamma for variance of lambda (gamma is the second summand in the variance equation)
        tuning_squared = dot(tuning_curves_reshaped(:, ibin), tuning_curves_reshaped(:, ibin));
        gamma(:, ibin) = exp(rho_g) * dt^2 * sum(tuning_squared .* (K_g - 1), 'all'); 
        
        % % compute mean of gain
        % mean_gain(ibin, itrial) = sum(exp(1/2 * rho_g));

        % compute variance of gain
        % var_gain(:, ibin) = exp(rho_g) * (dt^2 / dt_bin^2) * (sum(K_g - 1, 'all'));
        var_gain(:, ibin) = ((dt^2 / dt_bin^2) * sum(K_g, 'all')) - 1;
    end

    var_count = mean_count + gamma;
end
