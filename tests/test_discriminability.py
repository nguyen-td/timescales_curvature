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
data_path = Path('data')
# f_name = 'sim_0197.mat' # slow sensory-driven time scale
f_name = 'sim_0000.mat' # slow sensory-driven time scale
S = scipy.io.loadmat(Path(data_path) / f_name)['S']
S_list = [S[0, i] for i in range(S.shape[1])]  # convert to list of structs

mat2 = scipy.io.loadmat(Path('data') / 'bin_sizes_all.mat')
bin_sizes_all = mat2['bin_sizes_all'][0][12:-2] # only consider reasonable bin sizes

save_path = Path('data') / 'results'
save_path.mkdir(parents=True, exist_ok=True)

n_neurons = [2, 4, 8, 10, 20, 40, 50, 100]

# unpack data
saved_curvatures = np.zeros((len(n_neurons), len(bin_sizes_all)))
saved_discrim = np.zeros((len(n_neurons), len(bin_sizes_all)))
for ineuron, n_neuron in enumerate(n_neurons): 
    print(f'Runinng for population size index: {ineuron}')
    for ibin in range(len(bin_sizes_all)): # loop over different bin sizes
        print(f'Running for bin size index: {ibin}')
        # rate = S_list[ineuron]['lambda'][0, 0]
        # tuning_curves = S_list[ineuron]['tuning_curves'][0, 0]
        # gain = S_list[ineuron]['gain'][0, 0]
        rate = S_list[0]['lambda']
        tuning_curves = S_list[0]['tuning_curves']
        gain = S_list[0]['gain']

        n_neurons = rate.shape[0]
        n_trials = rate.shape[2]
        dt = S_list[0]['dt'][0][0]

        # create population response vectors; the number of bins determine the number of representational locations
        spike_train = generate_spikes(rate, dt)
        bin_length = int(bin_sizes_all[ibin] / dt.item())
        n_bins = int(spike_train.shape[1] // bin_length)
        spike_train_reshaped = np.reshape(spike_train, (n_neurons, n_bins, bin_length, n_trials)) # n_neurons x n_bins x bin_length x n_trials
        mean_spike_train = np.sum(spike_train_reshaped, axis=2) # n_neurons x n_bins x n_trials
        
        # create data consistent with sklearn convention
        X = mean_spike_train.transpose(1, 2, 0).reshape((-1, n_neurons)) # n_samples x n_features
        y = np.repeat(np.arange(n_bins), n_trials)

        # split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # initialize the SVM classifier with One-vs-One strategy
        clf = OneVsOneClassifier(LinearSVC(random_state=42))
        clf.fit(X_train, y_train)

        # this list perfectly aligns with the clf.estimators_
        class_pairs = list(combinations(range(len(clf.classes_)), 2))
        y_pred = clf.predict(X_test) # ensemble (final) classification

        # compute discriminability
        discrim_mat = np.zeros((n_bins, n_bins))
        count_mat = np.zeros((n_bins, n_bins))

        for idx, (i, j) in enumerate(class_pairs):
            class_i = clf.classes_[i]
            class_j = clf.classes_[j]
            
            # restrict myself to only those data points that belong to the correct ground truth class
            label_inds = np.where(y_test == class_i | (y_test == class_j))
            if label_inds[0].size == 0: # if no correct classification is found (list is empty)
                continue
            else:
                discrim_mat[i, j] = discrim_mat[j, i] = clf.score(X_test[label_inds], y_test[label_inds])
                count_mat[i, j] = count_mat[j, i] = len(y_test[label_inds])
            # discrim_mat[i, j] = discrim_mat[j, i] = clf.score(X_test, y_test)
            # count_mat[i, j] = count_mat[j, i] = len(y_test)
                
        np.fill_diagonal(discrim_mat, 0.5)

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

            # save 
            np.save(Path('data') / 'results' / f'{f_name.split('.')[0]}_curvs', saved_curvatures)
            np.save(Path('data') / 'results' / f'{f_name.split('.')[0]}_discrim', saved_discrim)
        except:
            print("Something went wrong.")

            # store results
            saved_curvatures[ineuron, ibin] = np.nan
            saved_discrim[ineuron, ibin] = np.nanmean(discrim_mat)

            # save 
            np.save(Path('data') / 'results' / f'{f_name.split('.')[0]}_curvs', saved_curvatures)
            np.save(Path('data') / 'results' / f'{f_name.split('.')[0]}_discrim', saved_discrim)
        finally:
            continue