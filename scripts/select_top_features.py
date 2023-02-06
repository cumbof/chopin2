#!/usr/bin/env python3
"""
Select top features
"""

__author__ = "Fabio Cumbo (fabio.cumbo@gmail.com)"
__version__ = "0.1.0"
__date__ = "Feb 4, 2023"

import argparse as ap
import math
import os


def read_params():
    """
    Read and test input arguments

    :return:    The ArgumentParser object
    """

    p = ap.ArgumentParser(
        prog="chopin2",
        description="Select top features",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--basepath",
        type=os.path.abspath,
        required=True,
        help="Path to the folder with random seeds",
    )
    p.add_argument(
        "--top",
        type=float,
        required=False,
        help="Select the top X percentage features",
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

    # Count how many times a feature has been selected
    selections = dict()

    # Iterate over seeds
    seeds = 0.0
    for seed in os.listdir(args.basepath):
        seedpath = os.path.join(args.basepath, seed)

        # Search for the last run
        lastrun = -1
        for run in os.listdir(seedpath):
            runid = int(run.replace("run", ""))
            if runid > lastrun:
                lastrun = runid
        runpath = os.path.join(seedpath, "run{}".format(lastrun))

        # Last run contains one chunk only
        chunkdir = os.path.join(runpath, "chunk_0")
        with open(os.path.join(chunkdir, "selection.txt")) as sel:
            for line in sel:
                line = line.strip()
                if line:
                    if not line.startswith("#"):
                        if line not in selections:
                            selections[line] = 0
                        # Read selected features and count
                        selections[line] += 1
        
        # Count seeds
        seeds += 1.0
    
    # Compute how many features must be reported based on --top
    report = math.ceil(seeds*args.top/100.0) if args.top else report len(selections)

    # Sort features in descending order based on the number of occurrences
    count_top = 0
    for s in reversed(sorted(selections.keys(), key=lambda s: selections[s])):
        if count_top >= report:
            break
        print("{}\t{}/{}".format(s, selections[s], seeds))
        count_top += 1


if __name__ == "__main__":
    main()
