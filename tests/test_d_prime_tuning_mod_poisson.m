% Test discriminability from linear tuning curve for modulated Poisson process.

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
sigma_G = [0.1, 1, 10]; 
min_rate = 20;
max_rate = 100; % 0-100 spk/s over T seconds

figure(1)
colors = lines(numel(sigma_G)); % distinct colors for each sigma
t = tiledlayout(4, 3, 'TileSpacing', 'loose', 'Padding', 'compact');
for ibin = 1:numel(time_bins)
    time_bin_vec = linspace(0, T, T / time_bins(ibin));
    linear_rate = linspace(min_rate, max_rate, T / time_bins(ibin));

    % initialize d_prime and snr_transformed arrays
    d_prime = zeros(numel(sigma_G), numel(linear_rate));
    snr_transformed = zeros(numel(sigma_G), numel(linear_rate));
    
    % plot sdt_estimate and snr_transformed (estimates of d-prime)
    nexttile
    hold on, box off, axis square, grid on
    h = gobjects(numel(sigma_G),1); % store SDT handles
    for iSigma = 1:numel(sigma_G)

        % compute d_prime
        % mean_count = time_bin_vec .* rate .* time_bins(ibin);
        mean_count = linear_rate .* time_bins(ibin);
        var_count = mean_count + sigma_G(iSigma)^2 .* mean_count.^2;
        d_prime(iSigma, :) = (mean_count - mean_count(1)) ./ sqrt((var_count + var_count(1)) / 2);

        % compute transformed SNR
        % transformed_rate = 2 / sigma_G(iSigma) * asinh(sigma_G(iSigma) * sqrt(time_bin_vec .* rate .* time_bins(ibin)));
        transformed_rate = 2 / sigma_G(iSigma) * asinh(sigma_G(iSigma) * sqrt(linear_rate * time_bins(ibin)));
        snr_transformed(iSigma, :) = transformed_rate - transformed_rate(1);

        % plotting per sigma
        h(iSigma) = plot(time_bin_vec, d_prime(iSigma, :), 'Color', colors(iSigma, :), 'LineWidth', 1.4); % store ONLY this

        plot(time_bin_vec, snr_transformed(iSigma, :), '--', 'Color', colors(iSigma,:), 'LineWidth', 1.4)
    end
    legend(h, arrayfun(@(s) sprintf('\\sigma_G = %.2f', s), sigma_G, 'UniformOutput', false), 'Location', 'westoutside')
    title(['\Deltat = ' num2str(time_bins(ibin))])
    xlabel('Time (s)')
    ylabel("d'")
    
    % plot sdt_estimate / snr_transformed
    nexttile
    hold on, box off, axis square, grid on
    for iSigma = 1:numel(sigma_G)
        plot(time_bin_vec, d_prime(iSigma, :) ./ snr_transformed(iSigma, :), 'Color', colors(iSigma,:), 'LineWidth', 1.4)
    end
    title('SDT estimate / SNR transformed', 'Units', 'normalized', 'Position', [0.5 1.03 0])
    xlabel('Time (s)')
    axis([0 2 0 2])

    % plot sdt_estimate - snr_transformed
    nexttile
    hold on, box off, axis square, grid on
    for iSigma = 1:numel(sigma_G)
        plot(time_bin_vec, d_prime(iSigma, :) - snr_transformed(iSigma, :), 'Color', colors(iSigma,:), 'LineWidth', 1.4)
    end
    title('SDT estimate - SNR transformed', 'Units', 'normalized', 'Position', [0.5 1.03 0])
    xlabel('Time (s)')
    axis([0 T -2 2])
end
sgtitle('Solid = SDT (d''), Dashed = SNR (transformed)')

figure(2)
plot(time_bin_vec, linear_rate, 'LineWidth', 1.4)
title(['Simulated linear firing rate from ' num2str(min_rate) ' - ' num2str(max_rate) ' spikes/s'], 'FontSize', 15)
xlabel('Time (s)')
ylabel('Firing rate (spikes/s)')
axis square, grid on, box off