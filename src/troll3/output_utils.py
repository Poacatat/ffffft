import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def print_output(name, output_dir, magnitude=None, freqs=None, mean_magnitude=None, std_magnitude=None, start_freqs=None, band_area=None, mean_band_area=None, std_band_area=None):
    if magnitude is not None:
        if mean_magnitude is None:
            mean_magnitude = np.mean(magnitude)
        if std_magnitude is None:
            std_magnitude = np.std(magnitude)
        
        magnitude_threshold = mean_magnitude + std_magnitude*10

        if max(magnitude) > magnitude_threshold:
            print(f"Anomaly detected in {name}!")
            print(f"Max Magnitude: {max(magnitude)}")
            print(f"Mean Magnitude: {mean_magnitude}")
            print(f"Std Magnitude: {std_magnitude}")
            print(f"Anomaly Beacon might have called back with : {1/freqs[np.argmax(magnitude)]}s")   


        Path.mkdir(Path(output_dir, f"{name}/images"), exist_ok=True, parents=True)
        plt.figure()
        plt.axhline(mean_magnitude+std_magnitude*10, color='r', linestyle='--', label='Mean')
        plt.plot(freqs, magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Frequency Domain (FFT)")
        plt.grid(True)
        plt.savefig(Path(output_dir, f"{name}/images/frequency_domain.png"))
        plt.close()

        plt.figure()   

    if band_area is not None:
        band_area_threshold = mean_band_area + std_band_area*2.5

        if max(band_area) > band_area_threshold:
            print(f"Anomaly detected in {name} based on sliding band analysis!")
            print(f"Max Band Area: {max(band_area)}")
            print(f"Mean Band Area: {mean_band_area}")
            print(f"Std Band Area: {std_band_area}")    
            print(f"Anomaly Beacon might have called back with : {1/(start_freqs[np.argmax(band_area)] + (start_freqs[1]-start_freqs[0])/2)} s")



        Path.mkdir(Path(output_dir, f"{name}/images"), exist_ok=True, parents=True)
        plt.figure()
        plt.axvline(start_freqs[np.argmax(band_area)], color='b', linestyle='--', label='Max')
        plt.axhline(mean_band_area+std_band_area*2.5, color='r', linestyle='--', label='Mean')
        plt.plot(start_freqs, band_area)
        plt.xlabel("Window Start Frequency (Hz)")
        plt.ylabel("Area Under Spectrum")
        plt.title("Sliding Frequency Band Area")
        plt.grid(True)
        plt.savefig(Path(output_dir, f"{name}/images/sliding_band_area.png"))
        plt.close()
   
    #print(1/(start_freqs[np.argmax(band_area)]))