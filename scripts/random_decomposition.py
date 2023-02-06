#!/usr/bin/env python3
"""
Decompose a table into chunks to run chopin2 in parallel
"""

__author__ = "Fabio Cumbo (fabio.cumbo@gmail.com)"
__version__ = "0.1.0"
__date__ = "Jan 31, 2023"

import argparse as ap
import os

from pathlib import Path

import numpy as np
import pandas as pd


def read_params():
    """
    Read and test input arguments

    :return:    The ArgumentParser object
    """

    p = ap.ArgumentParser(
        prog="chopin2",
        description="Dataset decomposition for chopin2",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=os.path.abspath,
        required=True,
        help="Path to the input dataset",
    )
    p.add_argument(
        "--fieldsep",
        type=str,
        default="\t",
        help="Field separator",
    )
    p.add_argument(
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Path to the output folder",
    )
    p.add_argument(
        "--chunks",
        type=int,
        default=10,
        help="Split the input dataset in chunks",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for shuffling features",
    )
    p.add_argument(
        "--selections",
        type=os.path.abspath,
        default=None,
        help="Path to a previous run folder with chunks and chopin2 selections",
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version="\"chopin2\" version {} ({})".format(__version__, __date__),
        help="Print the current \"chopin2\" version and exit",
    )
    return p.parse_args()


def main():
    # Load command line parameters
    args = read_params()

    # Load the input dataset into a Pandas DataFrame
    dataset = pd.read_csv(args.dataset, sep=args.fieldsep)

    # Get columns
    columns = list(dataset.columns)

    # Save first and last column IDs
    ids = columns[0]
    classes = columns[-1]

    # Remove first and last columns from columns list
    columns = columns[1:-1]

    # Load selected features from previous runs
    selected_features = list()
    if args.selections:
        if os.path.isdir(args.selections):
            gen = Path(args.selections).glob("**/selection.txt")
            for filepath in gen:
                features = [line.strip() for line in open(str(filepath)).readlines() if line.strip() and not line.startswith("#")]
                selected_features.extend(features)
    
    # Reshape the dataset with the selected features only
    if selected_features:
        columns = list(set(columns).intersection(set(selected_features)))

    # Shuffle columns inplace to group them randomly
    np.random.RandomState(seed=args.seed).shuffle(columns)

    # Groups columns
    count = 0
    for i in range(0, len(columns), args.chunks):
        subgroup = [ids]+columns[i:i+args.chunks]+[classes]
        subdataset = dataset[subgroup]

        # Define the output folder
        chunkdir = os.path.join(args.outdir, "chunk_{}".format(count))
        if not os.path.isdir(chunkdir):
            os.makedirs(chunkdir, exist_ok=True)
        
        # Dump subdataset
        subdataset.to_csv(os.path.join(chunkdir, "dataset_{}.txt".format(count)), index=None, sep=args.fieldsep)

        # Increment chunk count
        count += 1


if __name__ == "__main__":
    main()
