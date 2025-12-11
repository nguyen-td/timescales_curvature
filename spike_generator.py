import numpy as np

def generate_spikes(rate, dt):
    """
    Generate spikes from an inhomogeneous Poisson process.

    Inputs:
    -------
    rate: (n_neurons x n_timepoints x n_trials) Numpy array
        Rate function (mean firing rate over time). 
    dt: Scalar
        Time interval between two time steps.

    Output:
    -------
    spike_train: (n_neurons x n_timepoints x n_trials) Numpy array 
        Spike train consisting of True/False (0 or 1) spike occurences
    """
    
    n_neurons = rate.shape[0]
    T = rate.shape[1]
    n_trials = rate.shape[2]
    x = np.random.uniform(0, 1, size=(n_neurons, T, n_trials))

    # generate spike trains
    spike_train = (x <= rate * dt)

    return spike_train