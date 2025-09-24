%% Sanity check mean and variance

clear all
clc

%% Parameterize time series

T = 2; % in seconds
dt = 0.01; % time bin in seconds

x = linspace(0, T, T / dt);

% compute mean using sliding window
mean_1 = 0;
for i=1:length(x)
    mean_1 = mean_1 + x(i); 
end
mean_1 = mean_1 / length(x);
mean_2 = sum(x .* dt) / T;
mean_3 = mean(x);