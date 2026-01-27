import argparse
from pathlib import Path
import sys
from .analysis import _full_analysis
from .output_utils import print_output


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
        "--output-dir",
        type=Path,
        required=False,
        default=None,
        help="Path to output directory",
    )
    parser.add_argument(
        "--timeline-stitching",
        action="store_true",
        default=False,
        help="Stitch timeline gaps before analysis",
    )
    return parser

def main(argv):
    parser = argument_parser()

    args = parser.parse_args(argv[1:])

    magnitude, freqs, mean_magnitude, std_magnitude, start_freqs, band_area,  mean_band_area, std_band_area =_full_analysis(args.input,args.timeline_stitching)
    print("Analysis complete, generating output...")
    print_output(
        name = args.name, output_dir = args.output_dir,

        magnitude = magnitude, freqs       = freqs,
        band_area = band_area, start_freqs = start_freqs, 

        mean_magnitude = mean_magnitude, std_magnitude = std_magnitude,
        mean_band_area = mean_band_area, std_band_area = std_band_area
    )

   




if __name__ == "__main__":
    main(sys.argv)