# Hyperdimensional Computing Classifier
#### Originally forked from [https://github.com/moimani/HD-Permutaion](https://github.com/moimani/HD-Permutaion)

This repository includes some Python 3 utilities to build a Hyperdimensional Computing classification model according to the architecture
originally introduced in [https://doi.org/10.1109/DAC.2018.8465708](https://doi.org/10.1109/DAC.2018.8465708)

It also includes a script able to correctly format the BRCA, KIRP, and THCA DNA-Methylation data from the paper 
[Classification of Large DNA Methylation Datasets for Identifying Cancer Drivers](https://doi.org/10.1016/j.bdr.2018.02.005) 
by Fabrizio Celli, Fabio Cumbo, and Emanuel Weitschek.

Due to the size of the datasets, they have not been reported on this repository but can be retrieved from [ftp://bioinformatics.iasi.cnr.it/public/bigbiocl_dna-meth_data/](ftp://bioinformatics.iasi.cnr.it/public/bigbiocl_dna-meth_data/).

The `isolet` dataset is part of the first forked version of the repository and it has been maintained in order to provide a simple 
toy model for testing purposes.

### Usage

```
python hdclass.py [--dataset [INPUT_PKL_DATASET]]
                  [--dimensionality [HD_DIMENSION]]
                  [--levels [HD_LEVELS]]
                  [--retrain [RETRAINING_ITERATIONS]]

optional arguments:
    --dimensionality [HD_DIMENSION]       default value 10000
    --retrain [RETRAINING_ITERATIONS]     default value 1
```