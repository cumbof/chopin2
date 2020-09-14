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
python hdclass.py [--dataset        [INPUT_MATRIX]          ]
                  [--fieldsep       [FIELD_SEPARATOR]       ]
                  [--training       [TRAINING_PERCENTAGE]   ]
                  [--seed           [REPRODUCIBILITY_SEED]  ]
                  [--pickle         [INPUT_PICKLE]          ]
                  [--dimensionality [HD_DIMENSION]          ]
                  [--levels         [HD_LEVELS]             ]
                  [--retrain        [RETRAINING_ITERATIONS] ]

optional arguments:
    --training [TRAINING_PERCENTAGE]      default value 80
    --seed [REPRODUCIBILITY_SEED]         default value 0
    --dimensionality [HD_DIMENSION]       default value 10000
    --retrain [RETRAINING_ITERATIONS]     default value 1

If the --pickle parameter is specified, --dataset, --fieldsep, and --training parameters will not be used
```

### Credits

Please credit our work in your manuscript by citing:

> Fabio Cumbo, Eleonora Cappelli, and Emanuel Weitschek, "A brain-inspired hyperdimensional computing approach for classifying massive DNA methylation data of cancer", MDPI Algoritms, 2020

> Fabio Cumbo and Emanuel Weitschek, "An in-memory cognitive-based hyperdimensional approach to accurately classify DNA-Methylation data of cancer", The 11th International Workshop on Biological Knowledge Discovery from Big Data (BIOKDD'20), Communications in Computer and Information Science, vol 1285. Springer, Cham, 2020 [https://doi.org/10.1007/978-3-030-59028-4_1](https://doi.org/10.1007/978-3-030-59028-4_1)
