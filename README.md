# intron_sortseq
Analysis pipeline and downstream processing for study titled "Control of gene output by intronic RNA structure". Includes processing of Sort-seq flow cytometry and sequencing data, as well as code to reproduce manuscript figures.

## Installing the package

We used Snakemake with a custom conda environment. In addition, if you want to start from raw data, the following software packages need to be installed and accessible for Snakemake: `fastp`, `bwa`, `Picard`, `LoFreq`, `samtools`. Set the path to each software package in the `config/snake_sort.yaml` file.

```bash
# clone repository
git clone https://github.com/lschaerfen/intron_sortseq.git
cd intron_sortseq

# create conda environment
conda env create --name sort_env --file=sort_env.yml # create new environment from template
conda activate sort_env # activate

# install python package
pip install -e .
```

## Re-generating analyses and figures from "Control of gene output by intronic RNA structure"
To re-generate the analyses, either run the analysis pipeline on deposited raw data, or download processed files from GEO accession number [GSE308590](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308590). Save the raw or processed data files in the `data` directory in the respective folder. If starting from raw data, execute the Snakemake pipeline:

```bash
conda activate CoST
snakemake -c 16 -d smk_rundir/run --configfile config/snake_sort.yaml --resources mem_mb=32000 --rerun-incomplete --use-conda
```

Analysis code for each individual figure is available in jupyter notebooks in the `notebooks` directory and can be executed directly using the processed data from GEO.
