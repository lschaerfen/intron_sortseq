import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm, rankdata
import random

def scale_norm_sort_mat(sort_mat, detect_cutoff_vec):
    contributions_seq_raw = np.sum(sort_mat>=detect_cutoff_vec, axis=0)
    contributions_seq = contributions_seq_raw/np.sum(contributions_seq_raw)

    # scale bin sequencing data based on contribution of each bin
    sort_mat_scaled = sort_mat*contributions_seq
    sort_mat_scaled = sort_mat_scaled/np.sum(sort_mat_scaled, axis=1)[:,None]

    return(sort_mat_scaled, contributions_seq)

def mle_gauss(centers, probs):

    result = minimize(neg_log_likelihood, [0.5, 0.1], args=(centers, probs), method='L-BFGS-B', bounds=[(-0.5, 1.5), (1e-6, 1)])
    mu_mle, sigma_mle = result.x
    
    bic = 2 * np.log(len(probs)) - 2 * -neg_log_likelihood([mu_mle, sigma_mle], centers, probs)

    return(mu_mle, sigma_mle, bic)

def neg_log_likelihood(params, x, y):
    mu, sigma = params
    log_likelihood = np.sum(y * np.log(norm.pdf(x, mu, sigma) + 1e-9))  # Add small epsilon to avoid log(0)
    return(-log_likelihood)

def get_dist(sort_mat):
    # Compute the correlation matrix
    ranked = np.apply_along_axis(rankdata, 0, sort_mat)
    corr = np.corrcoef(ranked, rowvar=False)
    return(corr)

def x_univar(n, spread=0.3):
    return(spread*(np.random.rand(n)-0.5))

def flier_mask(x):
    x = x[np.isfinite(x)]
    iqr = abs(np.percentile(x, 75) - np.percentile(x, 25))
    flier_up = np.percentile(x, 75) + 1.5*iqr
    flier_dn = np.percentile(x, 25) - 1.5*iqr
    mask = (x > flier_up) | (x < flier_dn)
    return(mask)

def random_mask(n, frac, seed=42):
    random.seed(seed)
    n_true = round(frac*n)
    msk = [True]*n_true + [False]*(n-n_true)
    random.shuffle(msk)
    return(np.array(msk))


