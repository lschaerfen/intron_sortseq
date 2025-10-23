## load locations of tools from config file (just for readability of bash code)
SAMTOOLS=config['SAMTOOLS']
PYTHON=config['PYTHON']
FASTP=config['FASTP']
STAR=config['STAR']

## collect sample information and gene lists for targeted analysis
import pandas as pd
import numpy as np

samples = pd.read_csv(config["SAMPLES"], delimiter=',').set_index('sample_name')
def get_output(samples, OUTPUTPATH):
    
    out_files = []
    out_files += [OUTPUTPATH + 'variants/' + i + '_vars.pkl' for i in list(samples.index)]
    return(out_files)   


## run rule
rule all:
    input:
        get_output(samples, config['OUTPUTPATH'])


## modules
rule fastp_merge:
    input:
        r1=config['DATAPATH'] + "{sample}_R1.fastq.gz",
        r2=config['DATAPATH'] + "{sample}_R2.fastq.gz"
    output:
        r=config['OUTPUTPATH'] + "fastq/{sample}_fp.fastq.gz",
        r1=config['OUTPUTPATH'] + "fastq/{sample}_R1_long_fp.fastq.gz",
        r2=config['OUTPUTPATH'] + "fastq/{sample}_R2_long_fp.fastq.gz",
        html=config['OUTPUTPATH'] + "fastq/{sample}_fastp.html",
        json=config['OUTPUTPATH'] + "fastq/{sample}_fastp.json"
    threads: 8
    params:
        l=config['MINREADLENGTH']
    shell:
        "{FASTP} -l {params.l} -c --trim_poly_g --merge --overlap_len_require 10 "
        "-w {threads} --json {output.json} --html {output.html} "
        "-i {input.r1} -I {input.r2} --merged_out {output.r} "
        "--out1 {output.r1} --out2 {output.r2}"

rule fastp:
    input:
        r1=ancient(config['DATAPATH'] + "{sample}_R1.fastq.gz"),
        r2=ancient(config['DATAPATH'] + "{sample}_R2.fastq.gz")
    output:
        r1=config['OUTPUTPATH'] + "fastq/{sample}_R1_fp.fastq.gz",
        r2=config['OUTPUTPATH'] + "fastq/{sample}_R2_fp.fastq.gz",
        html=config['OUTPUTPATH'] + "fastq/{sample}_fastp.html",
        json=config['OUTPUTPATH'] + "fastq/{sample}_fastp.json"
    threads: config['THREADS']
    params:
        l=config['MINREADLENGTH'],
    shell:
        "fastp -l {params.l} "
        "-w {threads} --json {output.json} --html {output.html} "
        "-i {input.r1} -I {input.r2} -o {output.r1} -O {output.r2}"
        

rule assign_variants:
    input:
        fastq=config['OUTPUTPATH'] + "fastq/{sample}_fp.fastq.gz",
        r1=config['OUTPUTPATH'] + "fastq/{sample}_R1_long_fp.fastq.gz",
        r2=config['OUTPUTPATH'] + "fastq/{sample}_R2_long_fp.fastq.gz"
    output:
        pkl=config['OUTPUTPATH'] + "variants/{sample}_vars.pkl"
    params:
        lib=config['LIB_FASTA'],
        read_len=config['READ_LEN']
    shell:
        "{PYTHON} ../../scripts/assign_variants.py --read_len {params.read_len} -m {input.fastq} -a {input.r1} -b {input.r2} -l {params.lib} -o {output.pkl}"