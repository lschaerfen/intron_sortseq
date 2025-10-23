import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from sort_src.antarna import AntHill
from sort_src.antarna import AntHill
import tempfile
import os
import xml.etree.ElementTree as ET
import shutil
from itertools import combinations
from scipy.spatial.distance import hamming
import subprocess

def antarna_design(Cseq, Cstr, noOfColonies, uniform_GC_sampling=True, temperature=30.0, min_hamming_dist=3):

    GC_range = reachableGC(Cseq)
    if uniform_GC_sampling:
        # sample GC content from uniform distribution
        tGC    = float(GC_range[0])
        tGCmax = float(GC_range[1])
    else:
        # set target GC to reachable GC
        tGC    = float(np.mean(GC_range))
        tGCmax = -1

    hill = AntHill()

    hill.params.modus = "MFE"
    hill.params.Cstr = Cstr
    hill.params.level = 1
    hill.params.tGCmax = tGCmax
    hill.params.tGCvar = -1
    hill.params.pseudoknots = False
    hill.params.pkprogram = False
    hill.params.pkparameter = False
    hill.params.HotKnots_PATH = ""
    hill.params.strategy = "A"
    hill.params.Cseq = Cseq
    hill.params.tGC = tGC
    hill.params.temperature = temperature
    hill.params.paramFile = ""
    hill.params.noGUBasePair = False
    hill.params.noLBPmanagement = True
    hill.params.noOfColonies = noOfColonies
    hill.params.output_file = ""
    hill.params.py = True
    hill.params.name = "antaRNA"
    hill.params.verbose = False
    hill.params.output_verbose = False
    hill.params.seed = 69
    hill.params.improve_procedure = "s"
    hill.params.Resets = 10
    hill.params.ants_per_selection = 10
    hill.params.ConvergenceCount = 130
    hill.params.antsTerConv = 100
    hill.params.alpha = 1
    hill.params.beta = 1
    hill.params.ER = 0.2
    hill.params.Cstrweight = 0.5
    hill.params.Cgcweight = 5.0
    hill.params.Cseqweight = 1.0
    hill.params.omega = 2.23
    hill.params.time = 600
    hill.params.plot = False

    hill.params.check()


    if hill.params.error == "0":
        hill.swarm()
    else:
        print(hill.params.error)

    # filter result such that we retain only sequences of certain minimum hamming distance to the others
    designs = [i[1].lstrip('Rseq:') for i in hill.result]
    designs = optimal_set_min_hamming(designs, min_hamming_dist)

    return(designs)


def optimal_set_min_hamming(seqs, min_hamm):
    
    def _is_valid_set(seqs, min_hamm):
        for seq1, seq2 in combinations(seqs, 2):
            if hamming(list(seq1), list(seq2))*len(seq1) < min_hamm:
                return(False)
        return(True)

    seqs_opt = seqs.copy()
    while not _is_valid_set(seqs_opt, min_hamm):
        hamm_avg = [np.mean([hamming(list(i), list(j))*len(i) for j in seqs_opt[:idx] + seqs_opt[idx+1:]]) for idx, i in enumerate(seqs_opt)]
        seqs_opt.pop(np.argmin(hamm_avg))

    return(seqs_opt)


def reachableGC(seq):
    """
        Checks if a demanded GC target content is reachable in dependence with the given sequence constraint.
        For each explicit and ambiguous character definition within the constraint, the respective possibilities
        are elicited: "A" counts as "A", but for "B", also an "U" is possible, at the same time, "G" or "C" are possible
        as well. So two scenarios can be evaluated: A minimum GC content, which is possible and a maximum GC content.
        For this, for all not "N" letters their potential towards  their potentials is evaluated and counted in the
        respective counters for min and max GC content. Only those characters are taken into concideration which would enforce
        a categorial pyrimidine/purine decision. (ACGU, SW)
        
    """

    nucleotide_contribution = 1/float(len(seq)) 
    
    minGC = 0.0
    maxGC = 1.0
    for i in seq:
        if i != "N":
            if i == "A" or i == "U":
                maxGC -= nucleotide_contribution
            elif i == "C" or i == "G":
                minGC += nucleotide_contribution
            elif i == "S":#(G or C)
                minGC += nucleotide_contribution
            elif i == "W":#(A or T/U):
                maxGC -= nucleotide_contribution
    return([minGC, maxGC])

def rf_mutate(seq, dbr, motif_start, n_muts, n_vars=2, destab_side='right', RNAFramework_path='/Users/leo/Documents/repos/RNAFramework', ViennaRNA_path='/Users/leo/Builds/ViennaRNA/bin/RNAfold', perl5_lib='/Users/leo/Builds/ViennaRNA/lib/perl5/site_perl/5.38/'):
    """
    Run rf_mutate from RNAFramework package to design structure mutants.

    Parameters:
        seq (str): RNA sequence.
        dbr (str): Secondary structure in dot bracket format.
        motif_start (int): Start position of structure motif to disrupt. Must be at base of stem.
        n_muts (int): Number of mutations to design.
        RNAFramework_path (str): Path to RNAFramework executables.
        ViennaRNA_path (str): Path to ViennaRNA executables.

    Returns:
        ?
    """

    fd_mot, path_mot = tempfile.mkstemp(suffix='.csv')
    fd_seq, path_seq = tempfile.mkstemp(suffix='.fasta')
    out_dir          = tempfile.mkdtemp()

    # write the motif file for RNAFramework
    with os.fdopen(fd_mot, 'w') as f:
        f.write(f"{path_seq.split('/')[-1][:-6]};{motif_start}\n")

    # write the sequence for RNAFramework
    with os.fdopen(fd_seq, 'w') as f:
        f.write(f">{path_seq.split('/')[-1][:-6]}\n")
        f.write(f"{seq}\n")
        f.write(f"{dbr}\n")

    my_env = os.environ.copy()
    my_env['PERL5LIB'] = perl5_lib

    out = subprocess.run([RNAFramework_path + '/rf-mutate', '--vienna-rnafold', ViennaRNA_path, '-p', str(1), 
                          '-ow', '--output-dir', out_dir, '-nr', '-nm', str(n_muts), '-mf', path_mot, path_seq], 
                          capture_output=False, shell=False, stdout=open(os.devnull, 'wb'), env=my_env)

    os.remove(path_mot)
    os.remove(path_seq)

    # parse output
    tree = ET.parse(f"{out_dir}/{path_seq.split('/')[-1][:-6]}/motif_2-31.xml")
    shutil.rmtree(out_dir)

    root = tree.getroot()

    dgs = []
    mut = []
    for child in root:
        bases = [int(i) for i in child[0].attrib['bases'].split(',')]
        if destab_side == 'right' and all([True if i > 12 else False for i in bases]):
            dgs.append(float(child[0].attrib['ddG']))
            mut.append(child[0][0].text)
        elif destab_side == 'left' and all([True if i < 17 else False for i in bases]):
            dgs.append(float(child[0].attrib['ddG']))
            mut.append(child[0][0].text)

    if mut == []:
        return([])
    
    # sort results by ∆∆G
    order = np.argsort(dgs)[::-1]
    muts = [mut[i] for i in order]
    ddgs = [dgs[i] for i in order]

    # select destabilized seqs that have mutations in different positions (2 can overlap)
    ddg_thresh = ddgs[0]-(ddgs[0]/4)
    muts_final = [muts[0]]
    i = 1
    while len(muts_final) < n_vars and i < len(muts) and ddgs[i] >= ddg_thresh:
        if all([np.sum(np.array(list(j)) != np.array(list(k))) >= 2*n_muts-2 for j, k in combinations(muts_final+[muts[i]], 2)]):
            muts_final.append(muts[i])
        i += 1

    return(muts_final)


def get_destab_muts(seq_list, Cseqs, n_vars_destab, destab_side='right', **kwargs):

    assert len(Cseqs) == len(seq_list), "lists must be of same length"
    
    seqs_destab = []
    for i in tqdm(range(len(Cseqs))):
        des_pos = Cseqs[i].find('NNNNNNNNNNNNNNNNNNN')
        destab = []    
        for target_seq in seq_list[i]:

            if destab_side == 'right':
                seq = target_seq[des_pos-15:des_pos+19]
            elif destab_side == 'left':
                seq = target_seq[des_pos:des_pos+34]
                
            dbr = '..(((((((((((((....)))))))))))))..'
            motif_start = 2
            n_muts = 3
            
            muts = rf_mutate(seq, dbr, motif_start, n_muts, destab_side=destab_side, n_vars=n_vars_destab, **kwargs)

            if len(muts) == 0:
                print(f'no mutations found for seq {i}')
                destab.append([])
            else:
                
                if destab_side == 'right':
                    # put them back into the main sequence
                    destab.append([target_seq[:des_pos-13] + mut + target_seq[des_pos+17:] for mut in muts])

                if destab_side == 'left':
                    # put them back into the main sequence
                    destab.append([target_seq[:des_pos+2] + mut + target_seq[des_pos+32:] for mut in muts])
            
                if len(muts) < n_vars_destab:
                    print(f'only {len(muts)} mutation(s) found for seq {i}')

        seqs_destab.append(destab)
        
    return(seqs_destab)