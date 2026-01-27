import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def sliding_band_analysis(freqs: np.ndarray, magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 

    '''
    Computes the area under the magnitude curve for sliding frequency bands.
    :param freqs: The frequency array
    :param magnitude: The magnitudes
    :return: Description
    :rtype: tuple[ndarray, ndarray]
    '''

    window_width =  freqs.max()/10
    step = window_width / 100

    start_freqs = np.arange(0, freqs.max() - window_width, step)
    print(freqs.max()-window_width)
    band_area = []

    for f_start in start_freqs:
        f_end = f_start + window_width
        band_mask = (freqs >= f_start) & (freqs < f_end)

        area = np.trapezoid(magnitude[band_mask], freqs[band_mask])
        band_area.append(area)

    band_area = np.array(band_area)

    return start_freqs, band_area



def sliding_window_analysis(freqs, magnitude, window_width, step):
    pass

def timeline_stitching(timestamps, name, output_dir):
    

    df = pd.DataFrame({"timestamp": timestamps})

    df["dt"] = timestamps.diff().dt.total_seconds()


    #EXPECTED_DT = 1.0       # seconds (adjust to your data)
    #GAP_THRESHOLD = 10.0    # seconds
    EXPECTED_DT = df["dt"].median()
    #GAP_THRESHOLD = 170000 * EXPECTED_DT
    GAP_THRESHOLD = 60*60*1.5


    df["is_gap"] = df["dt"] > GAP_THRESHOLD
 
    # amount of time to remove at each step
    df["gap_duration"] = df["dt"].where(df["is_gap"], 0)

    adjusted = pd.Series(timestamps).copy()

    removed_time = 0.0

    for i in range(len(adjusted)):
        removed_time += df["gap_duration"].iloc[i]

        adjusted.iloc[i] = adjusted.iloc[i] - pd.Timedelta(seconds=removed_time)

    return adjusted


def _full_analysis(input: str, timeline_stitch: bool=False) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, float, float]:

    df = pd.read_csv(input)

    try:
        magnitude, freqs = fourier_analysis(df["timestamp"], timeline_stitch)
    except ValueError as e:
        print("Fourier analysis failed", file=sys.stderr)
        raise SystemExit    
    mean_magnitude = np.mean(magnitude)
    std_magnitude = np.std(magnitude)

    start_freqs,band_area = sliding_band_analysis(freqs, magnitude)
    mean_band_area = np.mean(band_area)
    std_band_area = np.std(band_area)
    return  magnitude, freqs, mean_magnitude, std_magnitude, start_freqs, band_area, mean_band_area, std_band_area


def fourier_analysis(timestamps: pd.Series, timeline_stitch=False) -> tuple[np.ndarray, np.ndarray]:   

    if timestamps.dtype != 'datetime64[ns, UTC]':
        try:
            timestamps = pd.to_datetime(timestamps, format='ISO8601').astype("datetime64[ns, UTC]")
        except ValueError as e:
            print("Error converting timestamps:", str(e), file=sys.stderr)
            print("Timestamps must be in datetime64[ns, UTC] format", file=sys.stderr)
            print("Aborting analysis.", file=sys.stderr)
            raise e
    if len(timestamps) < 3:
        print("Aborting analysis.", file=sys.stderr)
        raise ValueError("Not enough timestamps for analysis.", file=sys.stderr)
    
    timestamps = timestamps.sort_values().reset_index(drop=True)

    if timeline_stitch:
        timestamps = timeline_stitching(timestamps)
    
    t0 = timestamps.iloc[0]
    time_seconds = (timestamps - t0).dt.total_seconds()

    gaps = time_seconds.diff().dropna().values  

    if not np.all(gaps > 0): 
        print("Non-positive gap detected!")
        gaps = gaps[gaps > 0]

        
    dt = (np.mean(gaps)) /2    
     # time bin size in seconds 

    fs = 1 / dt 

    t_max = time_seconds.max()
    bins = np.arange(0, t_max + dt, dt)

    # Count events per bin
    signal, _ = np.histogram(time_seconds, bins=bins)

    signal = signal - np.mean(signal)


    n = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=dt)

    #positive frequencies only
    mask = freqs >= 0
    freqs = freqs[mask][1:]
    magnitude = np.abs(fft_vals[mask])[1:]



    return magnitude, freqs

   