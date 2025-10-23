import numpy as np
import pickle
from matplotlib.path import Path
from sort_src.flow_utils import load_flow, fit_gauss
from sort_src.pydscatter import dscatter_contour, dscatter_plot

class FlowData():

    def __init__(self, filename, ident, sign_chan='FITC-A', ctrl_chan='PE-A', norm=None):

        self.ident           = ident
        self.sign_chan       = sign_chan
        self.ctrl_chan       = ctrl_chan
        self.data, self.time = load_flow(filename)
        self.rat             = np.log2(self.data[self.sign_chan].to_numpy() / self.data[self.ctrl_chan].to_numpy())
        not_nan_mask         = self.rat==self.rat
        self.rat             = self.rat[not_nan_mask]
        self.data            = self.data.filter(not_nan_mask)
        self.gate            = np.ones(len(self.rat), dtype=bool)

        if norm is not None:
            self.rat_norm = (self.rat-norm[0]) / (norm[1]-norm[0])

    def load_gate(self, filename_gate, log_gate=False):
        
        with open(filename_gate, 'rb') as f:
            gate, chanx, chany = pickle.load(f)

        self.gate_coords = gate
        polygon_path = Path(self.gate_coords)

        x = self.data[chanx].to_numpy()
        y = self.data[chany].to_numpy()

        if log_gate:
            x = np.log10(x)
            y = np.log10(y)

        points = np.column_stack((x, y))
        contained = polygon_path.contains_points(points)

        self.gate = self.gate & contained

    def est_params(self, use_gate=True, norm=False, n=1):

        if norm:
            bins=np.arange(-0.3, 1.22, 0.005)
        else:
            bins=np.arange(-9, 5, 0.01)

        if n == 1:

            if norm:
                x, y, y_hat, mean_hat, std_hat, r2 = fit_gauss(self.rat_norm[self.gate], bins=bins)
                self.mean_norm = mean_hat
                self.std_norm  = std_hat
            else:
                x, y, y_hat, mean_hat, std_hat, r2 = fit_gauss(self.rat[self.gate], bins=bins)
                self.mean = mean_hat
                self.std  = std_hat
            
        elif n == 2:

            if norm:
                x, y, y_hat, amp1_hat, mean1_hat, std1_hat, mean2_hat, std2_hat, r2 = fit_gauss(self.rat_norm[self.gate], bins=bins, n=2)
            else:
                x, y, y_hat, amp1_hat, mean1_hat, std1_hat, mean2_hat, std2_hat, r2 = fit_gauss(self.rat[self.gate], bins=bins, n=2)
            
            if mean1_hat > mean2_hat:
                if norm:
                    self.mean_norm = mean1_hat
                    self.std_norm  = std1_hat
                else:
                    self.mean = mean1_hat
                    self.std  = std1_hat
            else:
                if norm:
                    self.mean_norm = mean1_hat
                    self.std_norm  = std1_hat
                else:
                    self.mean = mean2_hat
                    self.std  = std2_hat

        self.r2 = r2

    def scatter(self, axs, use_gate=True, show_gate=False, **kwargs):

        kwargs.setdefault('s', 2)
        kwargs.setdefault('alpha', 0.05)
        kwargs.setdefault('color', 'yellowgreen')

        x = np.log10(self.data[self.ctrl_chan])[self.gate] if use_gate else np.log10(self.data[self.ctrl_chan])
        y = np.log10(self.data[self.sign_chan])[self.gate] if use_gate else np.log10(self.data[self.sign_chan])

        axs.scatter(x, y, label=self.ident, edgecolors='none', **kwargs)
        if show_gate:
            axs.scatter(np.log10(self.data[self.ctrl_chan])[~self.gate], np.log10(self.data[self.sign_chan])[~self.gate], 2, c='r', alpha=0.05, edgecolors='none')
        axs.axis('equal')
        axs.set_xlim([1, 5])
        axs.set_ylim([1, 5])
        axs.set_xlabel('log10(red)')
        axs.set_ylabel('log10(green)')

        return(axs)

    def scatter_gate(self, axs, chanx, chany, dots=True, contour=True, use_gate=True, show_gate=False, dots_cmap='copper', contour_cmap='Greys',  **kwargs):

        kwargs.setdefault('alpha', 0.05)

        x = (self.data[chanx])[self.gate] if use_gate else self.data[chanx]
        y = (self.data[chany])[self.gate] if use_gate else self.data[chany]

        if dots:
            dscatter_plot(x, y, label=self.ident, edgecolors='none', markersize=10, cmap=dots_cmap, **kwargs)
        if contour:
            dscatter_contour(x, y, cmap=contour_cmap, alpha=0.5)
        if self.gate is not None and show_gate:
            axs.plot(self.gate_coords[:,0], self.gate_coords[:,1], '-k')

        # axs.axis('equal')
        axs.set_xlabel('log10(FSC)')
        axs.set_ylabel('log10(SSC)')


    def get_hist_ratio(self, norm=False, use_gate=True, bins=None):
        
        rat = self.rat[self.gate] if use_gate else self.rat

        if norm:
            if bins is None:
                bins=np.arange(-0.25, 1.25, 0.01)
            rat = self.rat_norm[self.gate] if use_gate else self.rat_norm

        if bins is None:
            bins = np.arange(-8, 4, 0.05)
        y, edges = np.histogram(rat, bins=bins, density=True)

        return(edges[:-1]+0.5*np.diff(edges)[0], y)