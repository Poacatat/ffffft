import pandas as pd
import numpy as np
import sys
from typing import Union
from scipy.stats import weibull_min

def sliding_band_analysis(freqs: np.ndarray, magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 

    '''
    Computes the area under the magnitude curve for sliding frequency bands.
    :param freqs: The frequency array
    :param magnitude: The magnitudes
    :return: Description
    :rtype: tuple[ndarray, ndarray]
    '''

    if len(freqs) != len(magnitude):
        raise ValueError("Frequency and magnitude arrays must be of the same length.")

    if len(freqs) < 10:
        return freqs, magnitude

    window_width =  freqs.max()/10
    step = window_width / 100

    start_freqs = np.arange(0, freqs.max() - window_width, step)
    band_area = []

    for f_start in start_freqs:
        f_end = f_start + window_width
        band_mask = (freqs >= f_start) & (freqs < f_end)

        area = np.trapezoid(magnitude[band_mask], freqs[band_mask])
        band_area.append(area)

    band_area = np.array(band_area)

    return start_freqs, band_area

def bootstraping(timestamps: Union[pd.Series, np.ndarray], frequency: float, iterations: int=1):
     return fourier_analysis(timestamps, timeline_stitch=False, dt=1/(4*frequency))


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


def _full_analysis(input: str, timeline_stitch: bool=False, dt=None) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, float, float]:

    df = pd.read_csv(input)

    try:
        magnitude, freqs = fourier_analysis(df["timestamp"], timeline_stitch, dt=dt)
    except ValueError as e:
        print("Fourier analysis failed", file=sys.stderr)
        raise SystemExit    
    mean_magnitude = np.mean(magnitude)
    std_magnitude = np.std(magnitude)

    start_freqs,band_area = sliding_band_analysis(freqs, magnitude)
    mean_band_area = np.mean(band_area)
    std_band_area = np.std(band_area)

    weibull_p = weibull_analysis(magnitude)

    return  magnitude, freqs, mean_magnitude, std_magnitude, start_freqs, band_area, mean_band_area, std_band_area, weibull_p

def weibull_analysis(data: Union[pd.Series, np.ndarray]) -> float:
    ''' 
    Perform Weibull analysis on the given data.\n
    Determines the probability of observing the maximum value in the dataset based on the fitted Weibull distribution.
    '''
    
    if isinstance(data, np.ndarray):
        if data.dtype != float:
            data = data.astype(float)
        data = pd.Series(data)
    if data.dtype != float:
        try:
            data = data.astype(float)
        except ValueError as e:
            print("Error converting data to float:", str(e), file=sys.stderr)
            print("Data must be convertible to float for Weibull analysis", file=sys.stderr)
            print("Aborting analysis.", file=sys.stderr)
            raise SystemExit
        

    shape, loc, scale = weibull_min.fit(data)

    n = len(data)
    x_max = np.max(data)

    F_xmax = weibull_min.cdf(x_max, shape, loc=loc, scale=scale)
    # ^ the probabilty of a single value being greater than x_max

    p_max = 1 - (F_xmax ** n)
    # ^ the probability of at least one value in n samples being greater than x_maxs
    return p_max


def fourier_analysis(timestamps: Union[pd.Series, np.ndarray], timeline_stitch: bool=False, dt=None, de_trend: bool=True) -> tuple[np.ndarray, np.ndarray]:   
    
    
    if isinstance(timestamps, np.ndarray):
        timestamps = pd.Series(timestamps)
    
    #if timestamps.dtype != 'datetime64[ns, UTC]':
    try:
        timestamps = pd.to_datetime(timestamps, utc=True)#.astype("datetime64[ns, UTC]")
    except ValueError as e:
        print("Error converting timestamps:", str(e), file=sys.stderr)
        print("Timestamps must be in datetime64[ns, UTC] format", file=sys.stderr)
        print("Aborting analysis.", file=sys.stderr)
        raise e
    if len(timestamps) < 3:
        print("Aborting analysis.", file=sys.stderr)
        raise ValueError("Not enough timestamps for analysis.", file=sys.stderr)
    assert timestamps.dtype == "datetime64[ns, UTC]"
    timestamps = timestamps.sort_values().reset_index(drop=True)

    if timeline_stitch:
        timestamps = timeline_stitching(timestamps)
    
    t0 = timestamps.iloc[0]
    time_seconds = (timestamps - t0).dt.total_seconds()

    gaps = time_seconds.diff().dropna().values  

    if not np.all(gaps > 0): 
        print("Non-positive gap detected!")
        gaps = gaps[gaps > 0]



    if dt is None:
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
  

    #positive frequencies only, and take away the zero frequency
    mask = freqs >= 0
    freqs = freqs[mask][1:]
    magnitude = np.abs(fft_vals[mask])[1:]

    if de_trend:
        total_length = t_max - time_seconds.min()
        threshold_freq = 3.5/total_length
        detrend_mask = freqs >= threshold_freq
        magnitude = magnitude[detrend_mask]
        freqs = freqs[detrend_mask]



    return magnitude, freqs

   