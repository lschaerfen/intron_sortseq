#!/bin/env/python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import gzip
import time
import numpy as np
import argparse
import pickle
from Bio.Seq import reverse_complement
from Bio import SeqIO
from itertools import zip_longest

def load_fasta(fasta_file):
    variants = []
    for i in SeqIO.parse(fasta_file, 'fasta'):
        variants.append(str(i.seq))
    return(variants)

def parse_fastq(input_handle):
    fastq_iterator = (l.rstrip() for l in input_handle)
    for record in zip_longest(*[fastq_iterator] * 4):
        yield(record)

def count_variants_naive(fastq, variants, primer_len=20):
    
    assert fastq[-9:] == '.fastq.gz', "Input files must be fastq.gz files."
    
    t_start = time.time()
    variants = set(variants)

    counts = dict(zip(variants, [0]*len(variants)))
    orphans = 0
    with gzip.open(fastq, 'rt') as f:

        for read in parse_fastq(f):
            rseq = read[1][primer_len:-primer_len]
            
            if rseq in variants:
                counts[rseq] += 1
            else:
                orphans += 1

    t_end = time.time()
    print(f"Time elapsed: {(t_end-t_start)/60:.2f} min")

    return(counts, orphans)

def count_variants_naive_PE(fastq_R1, fastq_R2, variants, primer_len=20, read_len=151):
    
    assert fastq_R1[-9:] == '.fastq.gz', "Input files must be fastq.gz files."
    assert fastq_R2[-9:] == '.fastq.gz', "Input files must be fastq.gz files."
    
    t_start = time.time()

    # crop to read len minus primer
    variants_cropped = [var[:read_len-primer_len] + var[-(read_len-primer_len):] for var in variants]
    cropped_dict = dict(zip(variants_cropped, variants)) # retain original sequence and cropped identity
    variants_cropped = set(variants_cropped)
    

    # no two cropped seqs can be the same
    assert len(variants_cropped) == len(variants), "At least two sequences cannot be unambiguously assigned."

    counts = dict(zip(variants, [0]*len(variants)))
    orphans = 0
    with gzip.open(fastq_R1, 'rt') as f,\
         gzip.open(fastq_R2, 'rt') as g:

        for read1, read2 in zip(parse_fastq(f), parse_fastq(g)):

            rseq = read1[1][primer_len:] + reverse_complement(read2[1][primer_len:])
            
            if rseq in variants_cropped:
                counts[cropped_dict[rseq]] += 1
            else:
                orphans += 1

    t_end = time.time()
    print(f"Time elapsed: {(t_end-t_start)/60:.2f} min")

    return(counts, orphans)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='merged FASTQ file to be processed.', type=str)
    parser.add_argument('-a', help='unmerged FASTQ file R1 to be processed.', type=str)
    parser.add_argument('-b', help='unmerged FASTQ file R2 to be processed.', type=str)
    parser.add_argument('-l', help='FASTA file containing the library sequences.', type=str)
    parser.add_argument('-o', help='Ouput file name.', type=str)

    # Adding two optional arguments with default values
    parser.add_argument('--primer_len', help='Length of the primer used for amplification.', type=int, default=20)
    parser.add_argument('--read_len', help='Sequencing read length (typically 151 or 101)', type=int, default=151)

    args = parser.parse_args()
    output_file = args.o
    fasta_file = args.l

    fastq = args.m
    fastq_R1 = args.a
    fastq_R2 = args.b
    variants = load_fasta(fasta_file)

    counts_m, orphans_m = count_variants_naive(fastq, variants)
    counts_p, orphans_p = count_variants_naive_PE(fastq_R1, fastq_R2, variants, primer_len=args.primer_len, read_len=args.read_len)

    # add together
    orphans = orphans_m + orphans_p
    counts = {k: counts_m.get(k, 0) + counts_p.get(k, 0) for k in counts_m.keys() & counts_p.keys()}

    n_ass = np.sum(list(counts.values()))
    print(f"{n_ass + orphans} reads processed. {100*n_ass/(n_ass+orphans):.1f}% assigned to variants.")

    with open(output_file, 'wb') as f:
        pickle.dump((counts, orphans), f)