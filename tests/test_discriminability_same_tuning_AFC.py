import scipy
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import h5py
from scipy import stats
import sys
import os
import torch
import mat73

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from spike_generator import generate_spikes

from perceptual_straightening.modules import ELBO

# load sample neural responses
# data_path = Path('data')
data_path = Path("C:/Users/tien/OneDrive - The University of Texas at Austin")
f_name = 'sim_0000_1k_trials.mat'
S = mat73.loadmat(Path(data_path) / f_name)

mat2 = scipy.io.loadmat(Path('data') / 'bin_sizes_all.mat')
bin_sizes_all = mat2['bin_sizes_all'][0][12:-2] # only consider reasonable bin sizes

save_path = Path('data') / 'results'
save_path.mkdir(parents=True, exist_ok=True)

n_neurons = [2, 4, 8, 10, 20, 40, 50, 100]
neuron_idx = 29

iscurvature = False # if true, compute curvature, else no

# unpack data
saved_curvatures = np.zeros((len(n_neurons), len(bin_sizes_all)))
saved_discrim = np.zeros((len(n_neurons), len(bin_sizes_all)))
for ineuron, n_neuron in enumerate(n_neurons): 
    print(f'Runinng for population size: {n_neuron}')
    for ibin in range(len(bin_sizes_all)): # loop over different bin sizes
        print(f'Running for bin size index: {ibin}')

        tuning_curves = S['S']['tuning_curves']
        gain = S['S']['gain']
        rate = np.repeat(tuning_curves[neuron_idx][np.newaxis, :], n_neuron, axis=0) * gain[:n_neuron]

        n_trials = rate.shape[2]
        dt = S['S']['dt']

        # create population response vectors; the number of bins determine the number of representational locations
        spike_train = generate_spikes(rate, dt)
        bin_length = int(bin_sizes_all[ibin] / dt.item())
        n_bins = int(spike_train.shape[1] // bin_length)
        spike_train_reshaped = np.reshape(spike_train, (n_neuron, n_bins, bin_length, n_trials)) # n_neurons x n_bins x bin_length x n_trials
        mean_spike_train = np.sum(spike_train_reshaped, axis=2) # n_neurons x n_bins x n_trials

        # get indices of all class/bin combinations
        class_list = list(combinations(np.arange(0, n_bins), 2))

        # compute discriminability
        discrim_mat = np.zeros((n_bins, n_bins))
        count_mat = np.zeros((n_bins, n_bins))
        for bin_comb in class_list:
            jbin = bin_comb[0]
            kbin = bin_comb[1]

            # create data consistent with sklearn convention
            X = mean_spike_train.transpose(1, 2, 0).reshape((-1, n_neuron)) # n_samples x n_features
            y = np.repeat(np.arange(n_bins), n_trials)

            # create training dataset with only two classes
            mask = np.isin(y, [jbin, kbin])
            X_train, X_test, y_train, y_test = train_test_split(X[mask], y[mask], test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # initialize the SVM classifier with 
            clf = SVC(kernel='linear', random_state=42)
            clf.fit(X_train, y_train)

            # store discriminability values
            discrim_mat[jbin, kbin] = discrim_mat[kbin, jbin] = np.sum(clf.predict(X_test) == y_test) / len(y_test)
            count_mat[jbin, kbin] = count_mat[kbin, jbin] = len(y_test)
                
        np.fill_diagonal(discrim_mat, np.nan)

        if iscurvature:
            try:
                # compute curvature
                n_corr_obs = discrim_mat * count_mat
                n_total_obs = count_mat

                n_dim = np.max([n_bins - 1, 2])
                n_iterations = 40000
                n_starts = 10

                elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=n_starts, n_iterations=n_iterations, verbose=True)
                c_est, p, elbo_loss_hist, kl_loss_hist, ll_loss_hist, c_prior_hist, d_prior_hist, l_prior_hist, c_post_hist, d_post_hist, l_post_hist = elbo.optimize_ELBO_SGD()

                # store results
                saved_curvatures[ineuron, ibin] = torch.rad2deg(torch.mean(c_est)).detach().numpy()
                saved_discrim[ineuron, ibin] = np.nanmean(discrim_mat)

                # # save 
                np.save(Path('data') / 'results' / f'{f_name.split('.')[0]}_curvs', saved_curvatures)
                np.save(Path('data') / 'results' / f'{f_name.split('.')[0]}_discrim', saved_discrim)
            except:
                print("Something went wrong.")

                # # store results
                saved_curvatures[ineuron, ibin] = np.nan
                saved_discrim[ineuron, ibin] = np.nanmean(discrim_mat)

                # # save 
                np.save(Path('data') / 'results' / f'{f_name.split('.')[0]}_curvs', saved_curvatures)
                np.save(Path('data') / 'results' / f'{f_name.split('.')[0]}_discrim', saved_discrim)
            finally:
                continue
        else:
            # store results
            saved_discrim[ineuron, ibin] = np.nanmean(discrim_mat)
            np.save(Path('data') / 'results' / f'{f_name.split('.')[0]}_discrim', saved_discrim)