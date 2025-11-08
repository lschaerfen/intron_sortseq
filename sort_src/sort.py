import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import polars as pl
from Bio import SeqIO
import matplotlib.pyplot as plt
from sort_src.flow import FlowData
from sort_src.sort_utils import scale_norm_sort_mat, mle_gauss
import pickle
from multiprocessing import Pool


class SORT():

    def __init__(self, csv_loc, fasta_loc, bin_size=0.01):

        self.fnames    = pl.read_csv(csv_loc)
        # self.variants  = pl.DataFrame({"index": [str(record.seq) for record in SeqIO.parse(fasta_loc, "fasta")]})
        self.variants = pl.DataFrame(dict(zip(["index", "Name"], zip(*[(str(r.seq), str(r.id)) for r in SeqIO.parse(fasta_loc, "fasta")]))))
        self.bin_size  = bin_size
        self.bins      = np.arange(-0.25, 1.25+bin_size, bin_size)

    def load_flow_data(self, plot=False):
        
        # load controls
        fnames_pos = (self.fnames
            .filter(pl.col("bin_ID").str.starts_with("pos"))
            .get_column("fcs")
            .to_list()
            )

        fnames_neg = (self.fnames
            .filter(pl.col("bin_ID").str.starts_with("neg"))
            .get_column("fcs")
            .to_list()
            )

        low_y = []
        low_t = []
        hig_y = []
        hig_t = []

        for fn_pos, fn_neg in zip(fnames_pos, fnames_neg):

            # negative control
            dum = FlowData(fn_neg, 'neg')
            dum.est_params()
            low_y.append(dum.mean)
            low_t.append(dum.time)
            
            # positive control
            dum = FlowData(fn_pos, 'pos')
            dum.est_params()
            hig_y.append(dum.mean)
            hig_t.append(dum.time)

        # linear fit control fluorescence over time for correction
        hig_slope, hig_intercept = np.polyfit(hig_t, hig_y, 1)
        low_slope, low_intercept = np.polyfit(low_t, low_y, 1)

        if plot:
            plt.figure(figsize=[2,2])
            plt.plot(hig_t, hig_y, 'ok')
            plt.plot(np.linspace(np.min(hig_t), np.max(hig_t), 100), hig_slope * np.linspace(np.min(hig_t), np.max(hig_t), 100) + hig_intercept, '--k')

            plt.plot(low_t, low_y, 'or')
            plt.plot(np.linspace(np.min(low_t), np.max(low_t), 100), low_slope * np.linspace(np.min(low_t), np.max(low_t), 100) + low_intercept, '--k')
            
        # load fluorescence data for each bin, correct according to timestamp
        fnames_bin = (self.fnames
                      .filter(pl.col('bin_ID').str.starts_with('bin'))
                      .get_column('fcs')
                      .to_list()
                      )

        bin_medians = []
        bin_hists = []
        for binx in fnames_bin:
            # print(binx)
            dum = FlowData(binx, 'bin')
            # estimate control fluorescence based on data collection time
            new_low = low_slope * dum.time + low_intercept
            new_hig = hig_slope * dum.time + hig_intercept

            # plot if desired
            if plot:
                plt.plot(dum.time, np.nanmedian(dum.rat), 'og')
            
            # normalize data btwn 0-1
            dum_norm = (dum.rat-new_low)/(new_hig-new_low)
            bin_median = np.nanmedian(dum_norm) 
            binned_y, _ = np.histogram(dum_norm, bins=self.bins, density=True)
            bin_hists.append(binned_y)
            bin_medians.append(bin_median)

        if plot:
            plt.xlabel('time (s)')
            plt.ylabel('fluorescence')
            plt.show()

        self.bin_medians = bin_medians
        self.bin_hists = bin_hists
        self.order = np.argsort(bin_medians)

    def add_sort_data(self, binx, fn_pos, fn_neg, pkl_loc, bin_id):

        # negative control
        dum = FlowData(fn_neg, 'neg')
        dum.est_params()
        low_y = dum.mean
        
        # positive control
        dum = FlowData(fn_pos, 'pos')
        dum.est_params()
        hig_y = dum.mean

        # bin data
        dum = FlowData(binx, 'bin')
        
        # normalize data btwn 0-1
        dum_norm = (dum.rat-low_y)/(hig_y-low_y)
        bin_median = np.nanmedian(dum_norm) 
        binned_y, _ = np.histogram(dum_norm, bins=self.bins, density=True)

        # add bins info to existing list
        self.bin_hists.append(binned_y)
        self.bin_medians.append(bin_median)

        # re-sort
        self.order = np.argsort(self.bin_medians)

        # add seq data location to csv
        new_row = pl.DataFrame({'bin_ID': [bin_id], 'fcs': ['/'], 'pkl': [pkl_loc]})
        self.fnames = pl.concat([self.fnames, new_row])


    def load_seq_data(self, min_reads=10, optimize_cutoff=False, cutoff=1, n_cpu=1):

        # load input library sequencing data
        with open(self.fnames.filter(pl.col("bin_ID") == "input")["pkl"][0], 'rb') as f:
            counts_dict, orphans = pickle.load(f)
        counts_tot = np.sum(list(counts_dict.values())) + len(counts_dict) # pseudocount added to avoid 0

        # print(f"input: {100*np.sum(counts_tot)/(np.sum(counts_tot)+orphans):.1f}% assigned")

        # normalize according to sequencing depth
        for seq in counts_dict:
            counts_dict[seq] = len(self.variants)*(counts_dict[seq]+1)/np.sum(counts_tot)
            
        sort_df = pl.DataFrame({"index": list(counts_dict.keys()), "input": np.array(list(counts_dict.values()), dtype=np.float64)})
        sort_df = sort_df.join(self.variants, on="index")

        # load sequencing data for each bin
        fnames_seq = self.fnames.filter(pl.col('bin_ID').str.starts_with('bin')).get_column('pkl').to_list()
        for idx, i in enumerate(self.order):
            with open(fnames_seq[i], 'rb') as f:
                counts_dict, orphans = pickle.load(f)

            counts_tot = np.sum(list(counts_dict.values()))
            
            # print(f"bin{idx+1}: {100*np.sum(counts_tot)/(np.sum(counts_tot)+orphans):.1f}% assigned")
            
            for seq in counts_dict:
                if counts_dict[seq] < min_reads: # read number filter
                    counts_dict[seq] = 0
                else:
                    counts_dict[seq] = len(self.variants)*counts_dict[seq]/np.sum(counts_tot)
                
            bin_df = pl.DataFrame({"index": list(counts_dict.keys()), f"bin{idx+1:02d}": np.array(list(counts_dict.values()), dtype=np.float64)})
            sort_df = sort_df.join(bin_df, on="index", how="inner")

        sort_mat = sort_df.select([f"bin{i+1:02d}" for i in range(len(fnames_seq))]).to_numpy()

        # here there is room for improvement: implement automatic choice of detection cutoff
        self.detect_cutoff = cutoff*np.ones(sort_mat.shape[1])
        
        sort_mat_scaled, contributions_seq = scale_norm_sort_mat(sort_mat, self.detect_cutoff)     
        bin_fluor_dist = np.array([self.bin_medians[i] for i in self.order], dtype=np.float64)

        # weighted average of bin fluorescence for each variant
        fluor_est = np.sum(sort_mat_scaled*bin_fluor_dist, axis=1)
        sort_df = sort_df.with_columns(pl.Series("fluor_est", fluor_est))

        # MLE estimate of mean and standard deviation
        args = [(bin_fluor_dist, y) for y in sort_mat_scaled]
        with Pool(processes=n_cpu) as pool:
            results = pool.starmap(mle_gauss, args)

        # Unzip the results into mu_est, si_est, and bic
        mu_est, si_est, bic = zip(*results)
        sort_df = sort_df.with_columns(pl.Series("mu_est", mu_est))
        sort_df = sort_df.with_columns(pl.Series("si_est", si_est))
        sort_df = sort_df.with_columns(pl.Series("bic", bic))

        self.sort_df  = sort_df
        self.sort_mat = sort_mat_scaled
        self.contributions_seq = contributions_seq

    def plot_hists(self, axs, scaled=False):
        if not scaled:
            for i in self.order:
                axs.bar(self.bins[:-1]+0.5*self.bin_size, self.bin_hists[i], self.bin_size, alpha=0.3)
                axs.plot(self.bins[:-1]+0.5*self.bin_size, self.bin_hists[i], color='gray', alpha=0.5)
        else:
            for idx, i in enumerate(self.order):
                axs.bar(self.bins[:-1]+0.5*self.bin_size, self.contributions_seq[idx]*self.bin_hists[i], self.bin_size, alpha=0.3)
                axs.plot(self.bins[:-1]+0.5*self.bin_size, self.contributions_seq[idx]*self.bin_hists[i], color='gray', alpha=0.5)

            axs.plot(self.bins[:-1]+0.5*self.bin_size, np.sum([self.contributions_seq[idx]*self.bin_hists[i] for idx, i in enumerate(self.order)], axis=0), '--k')

