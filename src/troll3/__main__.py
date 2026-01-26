import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from sliding_band import fourier_analysis


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Beacon fequecny analysis tool"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input data file",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        default=False,
        help="Disable caching mechanism",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        required=False,
        default=None,
        help="Path to cache file",
    )
    parser.add_argument(
        "--remove-zeroes",
        action="store_true",
        default=True,
        help="Removes all the zero-valued deltas from the dataset",
    )

    parser.add_argument(
        "--keep-zeroes",
        dest="remove_zeroes",
        action="store_false",
        help="Do not remove zero-valued deltas",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=False,
        default=None,
        help="Path to output directory",
    )
    return parser

def main(argv):
    parser = argument_parser()

    args = parser.parse_args(argv[1:])



    fourier_analysis(args)


    




if __name__ == "__main__":
    main(sys.argv)