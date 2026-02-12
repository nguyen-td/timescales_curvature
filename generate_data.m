%% Setup
clear all
clc

seed = 3;
rng(seed)

%% Simulate hierarchy of neural populations, copied from sim_hiearchy

n_frames = 11;        % number of video frames
frame_duration = 0.2; % duration over with a single frame was shown (seconds)
n_trials = 100;         % number of trials

% create time series
T = (n_frames * frame_duration);  % duration (seconds)
dt = 0.001;                       % simulation time steps (in s)
x = linspace(0, T, T / dt);       % dummy data points

% parameters of fast population
n_neurons = [20, 50, 80, 100];   % number of neurons within the population
rho_f = 3;                             % variance of tuning curves
tau_f = linspace(0.001, 2, 100);       % time scale of tuning curves

% parameters of shared gain of fast population
rho_g = 0.1;           % [min, max] variance of gain
tau_g = [0.002, 0.02]; % time scale of gain (seconds)
q_g = 2;               % power law exponent of gain covariance function
% tau_g = 0.02;          % time scale of gain (seconds)
% q_g = [0.5, 2];        % power law exponent of gain covariance function

%% Simulate neural responses
file_number = 1;

for itau = 1:numel(tau_f)
    disp(['Simulate for itau = ' num2str(itau)])
    for jtau = 1:numel(tau_g)
        disp(['Simulate for jtau = ' num2str(jtau)])
        clear S
        for ineuron = 1:numel(n_neurons)
            [tuning_curves, gain, lambda] = compute_CMP(n_trials, x, n_neurons(ineuron), rho_f, tau_f(itau), rho_g, tau_g(jtau), q_g);

            S{ineuron}.tuning_curves = tuning_curves;
            S{ineuron}.gain = gain;
            S{ineuron}.lambda = lambda;

            S{ineuron}.n_frames = n_frames;
            S{ineuron}.frame_duration = frame_duration;
            S{ineuron}.n_trials = n_trials;
            S{ineuron}.T = T;
            S{ineuron}.dt = dt;
            S{ineuron}.x = x;
            S{ineuron}.n_neuron = n_neurons(ineuron);
            S{ineuron}.rho_f = rho_f;
            S{ineuron}.tau_f = tau_f(itau);
            S{ineuron}.rho_g = rho_g;
            S{ineuron}.tau_g = tau_g(jtau);
            S{ineuron}.q_g = q_g;
        end
        save(fullfile('data', ['sim_' sprintf('%04d', file_number) '.mat']), 'S')
        file_number = file_number + 1;
        disp('---------------------------------')
    end
end
