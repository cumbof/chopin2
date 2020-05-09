# Hyperdimensional Computing Classifier
#### Originally forked from [https://github.com/moimani/HD-Permutaion](https://github.com/moimani/HD-Permutaion)

This repository includes some Python 3 utilities to build a Hyperdimensional Computing classification model according to the architecture
originally introduced in [https://doi.org/10.1109/DAC.2018.8465708](https://doi.org/10.1109/DAC.2018.8465708)

The `src/generators` folder contains two Python3 scripts able to create training a test datasets with randomly selected samples from:
- BRCA, KIRP, and THCA DNA-Methylation data from the paper [Classification of Large DNA Methylation Datasets for Identifying Cancer Drivers](https://doi.org/10.1016/j.bdr.2018.02.005) by Fabrizio Celli, Fabio Cumbo, and Emanuel Weitschek;
- Gene-expression quantification and Methylation Beta Value experiments provided by [OpenGDC](https://github.com/fabio-cumbo/OpenGDC/) for all the 33 different types of tumors of the TCGA program.

Due to the size of the datasets, they have not been reported on this repository but can be retrieved from: 
- [ftp://bioinformatics.iasi.cnr.it/public/bigbiocl_dna-meth_data/](ftp://bioinformatics.iasi.cnr.it/public/bigbiocl_dna-meth_data/)
- [http://geco.deib.polimi.it/opengdc/](http://geco.deib.polimi.it/opengdc/) and [https://github.com/fabio-cumbo/OpenGDC/](https://github.com/fabio-cumbo/OpenGDC/)

The `isolet` dataset is part of the original forked version of the repository and it has been maintained in order to provide a simple 
toy model for testing purposes only.

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
