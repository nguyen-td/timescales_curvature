% Test discriminability from linear tuning curve.

%% Setup
clear
close all
clc

%% Define tuning curve
% define parameters
T = 2;                      % duration (seconds)
dt = 1 / 1000;              % time bin (seconds)
max_freq = 100;             % maximum firing rate
sigma = [0.1, 1, 10];       % variance
time_bins = [0.01, 0.02, 0.05]; % time bins in seconds

% compute d_prime based on all possible noise combinations
all_sigma_combs = table2array(combinations(sigma, sigma)); 
rate = linspace(1, max_freq, T / dt);
d_primes_orig = compute_d_prime(T, time_bins, all_sigma_combs, rate);
d_primes_Olivier = compute_d_prime(T, time_bins, all_sigma_combs, rate, 'Olivier');

%% Plotting
figure(1)
plot(rate, 'LineWidth', 1.4)
title('Simulated firing rate')
xlabel('Time (ms)')
ylabel('Firing rate (spikes/s)')
set(gca,'FontSize', 12)
axis square
grid on

% original firing rates
figure(2)
max_y_lim = max(cell2mat(d_primes_orig), [], 'all');
tile = tiledlayout(1, numel(time_bins));
for ibin = 1:numel(time_bins)

    nexttile
    hold on
    for icomb = 1:size(all_sigma_combs, 1)
        plot(d_primes_orig{ibin}(icomb, :), 'LineWidth', 1.4, ...
            'DisplayName', ['\sigma^{2}_{1} = ' num2str(all_sigma_combs(icomb, 1)) '; \sigma^{2}_{2} = ' num2str(all_sigma_combs(icomb, 2))])
    end
    title(['d-prime for \Deltat = ' num2str(time_bins(ibin))])
    xlabel('Time bin (s)')
    ylabel('d-prime')
    ylim([0 max_y_lim])
    axis square
    grid on
    set(gca,'FontSize', 12)
    hold off
    if ibin == numel(time_bins)
        legend('Location', 'bestoutside')
    end
end

% Olivier's transformation
figure(3)
max_y_lim = max(cell2mat(d_primes_Olivier), [], 'all');
tile = tiledlayout(1, numel(time_bins));
for ibin = 1:numel(time_bins)

    nexttile
    hold on
    for icomb = 1:size(all_sigma_combs, 1)
        plot(d_primes_Olivier{ibin}(icomb, :), 'LineWidth', 1.4, ...
            'DisplayName', ['\sigma^{2}_{1} = ' num2str(all_sigma_combs(icomb, 1)) '; \sigma^{2}_{2} = ' num2str(all_sigma_combs(icomb, 2))])
    end
    title(['d-prime for \Deltat = ' num2str(time_bins(ibin))])
    xlabel('Time bin (s)')
    ylabel('d-prime')
    ylim([0 max_y_lim])
    axis square
    grid on
    set(gca,'FontSize', 12)
    hold off
    if ibin == numel(time_bins)
        legend('Location', 'bestoutside')
    end
end


function d_primes = compute_d_prime(T, time_bins, all_sigma_combs, rate, transform)
    % Input:
    % ------
    %   transform - [String] If Olivier's transformation should be used: 'Olivier'. If empty, no transformation on the rate will be used.
    
    if nargin < 5
        transform = 'none';
    end
    d_primes = {}; % n_time_bins x n_sigma_combs
    for ibin = 1:numel(time_bins)
        n_bins = T / time_bins(ibin);
        
        d_primes_diff = zeros(size(all_sigma_combs, 1), n_bins-1);
        for icomb = 1:size(all_sigma_combs, 1)
            sigma1 = all_sigma_combs(icomb, 1);
            sigma2 = all_sigma_combs(icomb, 2);
            
            binned_rates{ibin} = reshape(rate, [], n_bins);      % discretize rate (not really "bin" because we are dealing with analytical rate)
            if strcmpi(transform, 'Olivier')
                for idiff = 1:n_bins - 1
                    mus = binned_rates{ibin}(1, :);
                    y1 = 2 / sigma1 * asinh(sigma1 * sqrt(mus(idiff)));
                    y2 = 2 / sigma2 * asinh(sigma2 * sqrt(mus(idiff+1)));
                    d_primes_diff(icomb, idiff) = abs(y2 - y1) / sqrt((sigma1 + sigma2) / 2); 
                end
            else
                diff_mus = diff(binned_rates{ibin}(1, :));           % compute difference of means at the start of each bin, since bins are fixed all diff_mus are equal
                d_primes_diff(icomb, :) = diff_mus / sqrt((sigma1 + sigma2) / 2); 
            end
        end
        d_primes{ibin} = d_primes_diff;
    end
end