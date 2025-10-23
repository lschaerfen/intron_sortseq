import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
from sort_src.flow import FlowData
from matplotlib.widgets import PolygonSelector
import argparse
import pickle
from sort_src.pydscatter import dscatter_plot

def set_gate(fcs_file, output_name, chanx, chany):
    
    # FCS/SCC visualization and gating
    flow = FlowData(fcs_file, '')

    # show figure to draw gate
    fig, axs = plt.subplots(1, 1, figsize=[5, 5])

    dscatter_plot(flow.data[chanx], flow.data[chany], edgecolors='none', markersize=10, cmap='copper', alpha=1)

    axs.set_xlabel(chanx)
    axs.set_ylabel(chany)	

    selector = PolygonSelector(axs, lambda *args: None, useblit=True, props=dict(color='r', linestyle='-', linewidth=2, alpha=0.5))
    plt.show()

    gate = np.array(selector._xys)

    fig, axs = plt.subplots(1, 1, figsize=[5, 5])
    flow.scatter_gate(axs, chanx, chany, dots=True, contour=False, use_gate=False, show_gate=False)
    axs.plot(gate[:,0], gate[:,1], '-k')
    plt.show()

    with open(output_name, 'wb') as f:
        pickle.dump((gate, chanx, chany), f)

    print(f"Saved gate coordinates to file {output_name}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('x', help='Name of data for x axis. Options: [fsca, fscw, fsch, ssca, sscw, ssch]')
    parser.add_argument('y', help='Name of data for x axis. Options: [fsca, fscw, fsch, ssca, sscw, ssch]')
    parser.add_argument('fcs_file', help='.fcs file to use for setting gate')
    parser.add_argument('-o', help='Output file name.')
    args = parser.parse_args()

    if not args.o:
        args.o = args.fcs_file.rstrip('.fcs') + '.gate'

    header_dict = {'fsca':'FSC-A', 'fscw':'FSC-W', 'fsch':'FSC-H', 'ssca':'SSC-A', 'sscw':'SSC-W', 'ssch':'SSC-H'}

    set_gate(args.fcs_file, args.o, header_dict[args.x], header_dict[args.y])
