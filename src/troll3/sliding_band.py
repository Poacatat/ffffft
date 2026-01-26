import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def sliding_band_analysis(freqs, magnitude, args):
    window_width =  freqs.max()/10
    step = window_width / 1000

    start_freqs = np.arange(0, freqs.max() - window_width, step)
    print(freqs.max()-window_width)
    band_area = []

    for f_start in start_freqs:
        f_end = f_start + window_width
        band_mask = (freqs >= f_start) & (freqs < f_end)

        area = np.trapezoid(magnitude[band_mask], freqs[band_mask])
        band_area.append(area)

    band_area = np.array(band_area)


    plt.figure()
    plt.plot(start_freqs, band_area)
    plt.xlabel("Window Start Frequency (Hz)")
    plt.ylabel("Area Under Spectrum")
    plt.title("Sliding Frequency Band Area")
    plt.grid(True)
    plt.savefig(Path(args.output_dir, f"{args.name}/images/sliding_band_area.png"))


def timeline_stitching(timestamps, args):

    
    #plt.hist(timestamps, bins=250)
    #plt.xlabel("Index")
    #plt.ylabel("Time (s)")
    #plt.title("regulartime series")
    #plt.savefig(Path(args.output_dir, f"{args.name}/images/regular.png"))
    #plt.close()
    df = pd.DataFrame({"timestamp": timestamps})

    df["dt"] = timestamps.diff().dt.total_seconds()


    #EXPECTED_DT = 1.0       # seconds (adjust to your data)
    #GAP_THRESHOLD = 10.0    # seconds
    EXPECTED_DT = df["dt"].median()
    #GAP_THRESHOLD = 170000 * EXPECTED_DT
    GAP_THRESHOLD = 60*60*1.5


    df["is_gap"] = df["dt"] > GAP_THRESHOLD
 
    # Amount of time to remove at each step
    df["gap_duration"] = df["dt"].where(df["is_gap"], 0)

    adjusted = pd.Series(timestamps).copy()

    removed_time = 0.0

    for i in range(len(adjusted)):
        removed_time += df["gap_duration"].iloc[i]

        adjusted.iloc[i] = adjusted.iloc[i] - pd.Timedelta(seconds=removed_time)


    #plt.hist(adjusted, bins=250)
    #plt.xlabel("Index")
    #plt.ylabel("Adjusted Time (s)")
    #plt.title("Gap-removed time series")
    #plt.savefig(Path(args.output_dir, f"{args.name}/images/gap_removed_time_series.png"))
    

    return adjusted


def fourier_analysis(args):   

    df = pd.read_csv(args.input)
    timestamps = pd.to_datetime(df["timestamp"]).astype("datetime64[ns, UTC]")

    timestamps = timestamps.sort_values().reset_index(drop=True)

    timestamps = timeline_stitching(timestamps, args)    


    t0 = timestamps.iloc[0]
    time_seconds = (timestamps - t0).dt.total_seconds()

    gaps = time_seconds.diff().dropna().values  

    if not np.all(gaps > 0): 
        print("Non-positive gap detected!")
        gaps = gaps[gaps > 0]

        
    dt = (np.mean(gaps)) /2     # time bin size in seconds 

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


    mean_magnitude = np.mean(magnitude)
    std_magnitude = np.std(magnitude)

    print(f"Mean of magnitude: {mean_magnitude}")
    print(f"Standard deviation of magnitude: {std_magnitude}")
    # Plot the mean and standard deviation lines on time domain plot
    plt.figure()
    plt.axhline(mean_magnitude+std_magnitude*5, color='r', linestyle='--', label='Mean')

    plt.plot(freqs, magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain (FFT)")
    plt.grid(True)
    plt.savefig(Path(args.output_dir, f"{args.name}/images/frequency_domain.png"))


    sliding_band_analysis(freqs, magnitude, args)


