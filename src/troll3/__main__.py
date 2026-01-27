import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from sliding_band import _full_analysis
from matplotlib import pyplot as plt


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


    magnitude_threshold = mean_magnitude + std_magnitude*10
    band_area_threshold = mean_band_area + std_band_area*2.5

    if max(magnitude) > magnitude_threshold:
        print(f"Anomaly detected in {args.name}!")
        print(f"Max Magnitude: {max(magnitude)}")
        print(f"Mean Magnitude: {mean_magnitude}")
        print(f"Std Magnitude: {std_magnitude}")
        print(f"Anomaly Beacon called back with : {1/freqs[np.argmax(magnitude)]} s")

    if max(band_area) > band_area_threshold:
        print(f"Anomaly detected in {args.name} based on sliding band analysis!")
        print(f"Max Band Area: {max(band_area)}")
        print(f"Mean Band Area: {mean_band_area}")
        print(f"Std Band Area: {std_band_area}")
      

        print(f"Anomaly Beacon called back with : {1/(start_freqs[np.argmax(band_area)] + (start_freqs[1]-start_freqs[0])/2)} s")
    print(1/(start_freqs[np.argmax(band_area)]))


    plt.figure()
    plt.axhline(mean_magnitude+std_magnitude*5, color='r', linestyle='--', label='Mean')
    plt.plot(freqs, magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain (FFT)")
    plt.grid(True)
    plt.savefig(Path(args.output_dir, f"{args.name}/images/frequency_domain.png"))
    plt.close()

    plt.figure()

    plt.axvline(start_freqs[np.argmax(band_area)], color='b', linestyle='--', label='Max')
    plt.axhline(mean_band_area+std_band_area*2.5, color='r', linestyle='--', label='Mean')
    plt.plot(start_freqs, band_area)
    plt.xlabel("Window Start Frequency (Hz)")
    plt.ylabel("Area Under Spectrum")
    plt.title("Sliding Frequency Band Area")
    plt.grid(True)
    plt.savefig(Path(args.output_dir, f"{args.name}/images/sliding_band_area.png"))
    plt.close()




if __name__ == "__main__":
    main(sys.argv)