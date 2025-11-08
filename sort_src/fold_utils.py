import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import RNA
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from itertools import combinations, product

def sample_from_boltzman(seq, n, temp=30.0):
    md = RNA.md()
    md.temperature = temp
    md.uniq_ML = 1
    fc = RNA.fold_compound(seq, md)
    _, mfe = fc.mfe()
    # rescale Boltzmann factors according to MFE
    fc.exp_params_rescale(mfe)
    fc.pf()
    return([s for s in fc.pbacktrack(n)])

def vienna_all_metrics(seq, temp=30.0):
    md = RNA.md()
    md.temperature = temp
    fc = RNA.fold_compound(seq, md) # initialize
    dbr_efe, efe = fc.pf() # calculate partition function, returns mod. dbr notation and ensemble free energy
    ent = fc.positional_entropy() # entropy for each nucleotide
    dbr_mea, mea = RNA.MEA_from_plist(fc.plist_from_probs(0), seq) # maximum expected accuracy structure and ∆G
    dbr_mfe, mfe = fc.mfe() # minimum free energy structure and ∆G
    p_mfe = fc.pr_structure(dbr_mfe) # equilibrium probability of the MFE structure
    ed = fc.mean_bp_distance() # ensemble diversity; average distance between all structures in the ensemble
    ma = fc.bpp() # pairing probability matrix
    pp = np.sum(ma, axis=0) + np.sum(ma, axis=1) # pairing probability per nucleotide

    return(pp[1:], dbr_mea, mea, dbr_mfe, mfe, dbr_efe, efe, p_mfe, ent[1:], ed)

def vienna_bp_prob(seq, temp=30.0):
    md = RNA.md()
    md.temperature = temp
    fc = RNA.fold_compound(seq, md)
    fc.pf()
    ma = fc.bpp()
    return(ma)

def get_all_metrics(variants, n_cpu=1, temp=30.0):
    
    vienna = []
    vars_iterable = [(var, temp) for var in variants]
    with Pool(processes=n_cpu) as pool:
        for i in pool.starmap(vienna_all_metrics, vars_iterable):
            vienna.append(i)

    bpp, dbr_mea, mea, dbr_mfe, mfe, dbr_efe, efe, p_mfe, ent, ed = zip(*vienna)

    def _all_same_length(lst):
        return len({len(x) for x in lst}) <= 1
    
    if _all_same_length(bpp):
        return(np.array(bpp), np.array(dbr_mea), np.array(mea), np.array(dbr_mfe), np.array(mfe), np.array(dbr_efe), np.array(efe), np.array(p_mfe), np.array(ent), np.array(ed))
    else:
        return(bpp, dbr_mea, mea, dbr_mfe, mfe, dbr_efe, efe, p_mfe, ent, ed)

def get_coords(dbr, plot_type=4):

    RNA.cvar.rna_plot_type = plot_type
    coords = RNA.get_xy_coordinates(dbr)
    x = [coords.get(i).X for i, _ in enumerate(dbr)]
    y = [coords.get(i).Y for i, _ in enumerate(dbr)]
    return(x, y)
    
def get_partner(dbr):
    bpdict = {}
    opened = []
    for i, d in enumerate(dbr):
        if d == '(':
            opened.append(i)
        elif d == ')':
            partner = opened.pop(-1)
            if i not in bpdict:
                bpdict[i] = partner
                bpdict[partner] = i
    
    if opened:
        print('parentheses unmatched!')
        return(None)
    else:
        return(bpdict)
    
def get_distance(dbr):
    bpdict = get_partner(dbr)
    # add 0 for no pairs
    for i in range(len(dbr)):
        if i not in bpdict:
            bpdict[i] = i
    bp_dist = [abs(nt-bpdict[nt]) for nt in bpdict]
    order_bp = np.array(list(bpdict.keys())).argsort()
    bp_dist = np.array([bp_dist[i] for i in order_bp])
    return(bp_dist)

def filter_dbr_by_dist(dbr, max_dist):
    bp_dist = get_distance(dbr)
    dbr_f = np.array(list(dbr))
    dbr_f[bp_dist > max_dist] = '.'
    return(''.join(dbr_f))

def coords_to_segments(x, y):

    def _midpoint(p1, p2):
        x_mid = (p1[0] + p2[0]) / 2
        y_mid = (p1[1] + p2[1]) / 2
        return (x_mid, y_mid)

    x_mid = []
    y_mid = []

    for i, (x_i, y_i) in enumerate(zip(x[1:-1], y[1:-1])):

        x_m, y_m = _midpoint((x_i, y_i), (x[i], y[i]))
        x_mid.append(x_m)
        x_mid.append(x_i)
        y_mid.append(y_m)
        y_mid.append(y_i)

    return(x_mid, y_mid)

def plot_structure(seq, dbr, rea, mask, axs, text=True, circles=False, cmap='viridis', line_width=5, bp_width=2, circle_size=50, text_size=7, vmin=0, vmax=1, fontname='Input Mono'):
    
    assert len(seq) == len(dbr), "sequence and dot bracket must have same length"
    assert len(seq) == len(rea), "sequence and reactivity must have same length"
    assert len(seq) == len(mask), "sequence and mask must have same length"
    
    # get colors   
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    clrs = m.to_rgba(rea)
    clrs[~mask] = [230/255, 230/255, 230/255, 1]

    # get coordinates
    coords_x, coords_y = get_coords(dbr)

    if circles:
        axs.plot(coords_x, coords_y, '-k', zorder=0)
        axs.scatter(coords_x, coords_y, circle_size, color=clrs, edgecolors='none')
    else:
        # plot base pairs
        bp_dict = get_partner(dbr)
        done = []
        for pair in bp_dict:
            if set([pair, bp_dict[pair]]) not in done:
                
                # plot GU pairs as dot
                if set([seq[pair], seq[bp_dict[pair]]]) == set(['G', 'U']) or set([seq[pair], seq[bp_dict[pair]]]) == set(['G', 'T']):
                    
                    midpoint_x = (coords_x[pair] + coords_x[bp_dict[pair]]) / 2
                    midpoint_y = (coords_y[pair] + coords_y[bp_dict[pair]]) / 2
                    axs.scatter(midpoint_x, midpoint_y, 2*bp_width, color='k', edgecolors='none')

                # plot other pairs as line 
                else:    
                    if set([seq[pair], seq[bp_dict[pair]]]) == set(['G', 'C']):
                        lw = bp_width
                    elif set([seq[pair], seq[bp_dict[pair]]]) == set(['A', 'U']) or set([seq[pair], seq[bp_dict[pair]]]) == set(['A', 'T']):
                        lw = 0.75*bp_width
                    else:
                        lw = bp_width
                    axs.plot([coords_x[pair], coords_x[bp_dict[pair]]], [coords_y[pair], coords_y[bp_dict[pair]]], '-k', lw=lw)

                done.append(set([pair, bp_dict[pair]]))

        # plot the colored backbone
        x_mid, y_mid = coords_to_segments(coords_x, coords_y)
        for j in range(len(seq)-2):
            axs.plot([x_mid[2*j-2], x_mid[2*j-1], x_mid[2*j]], [y_mid[2*j-2], y_mid[2*j-1], y_mid[2*j]], color=m.to_rgba(rea[j]), linewidth=line_width, solid_capstyle='butt')

    # plot the nucleotide text
    if text:
        for i, nt in enumerate(seq.replace('T', 'U')):
            axs.text(coords_x[i], coords_y[i], nt, ha='center', va='center', color='w', size=text_size, fontname=fontname, fontweight='bold')
    axs.axis('equal')
    axs.axis('off')

    return(m)

def generate_point_mutations(full_stem, fw, n):
    bases = ['A', 'U', 'C', 'G']
    full_stem = list(full_stem)
    
    # Get indices of mutable positions
    n_indices = [i for i, c in enumerate(fw) if c == 'N']
    if n > len(n_indices):
        raise ValueError("n is greater than the number of mutable positions.")
    
    mutated_seqs = []

    # All combinations of n positions
    for pos_combo in combinations(n_indices, n):
        # For each position, determine the 3 alternative bases
        alt_bases_list = []
        for i in pos_combo:
            current_base = full_stem[i]
            alt_bases = [b for b in bases if b != current_base]
            alt_bases_list.append(alt_bases)
        
        # Cartesian product of alternative bases for all positions
        for replacements in product(*alt_bases_list):
            mutant = full_stem.copy()
            for i, new_base in zip(pos_combo, replacements):
                mutant[i] = new_base
            mutated_seqs.append(''.join(mutant))

    return(mutated_seqs)