import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import folium as fo
import numpy as np
import streamlit as st
import math
from scipy.signal import butter, filtfilt, welch, find_peaks
import scipy.signal
from  scipy.fft import fft, fftfreq


acceleration_data = pd.read_csv("Linear Accelerometer.csv")
location_data = pd.read_csv("Location.csv")

#Location and path plotting start
latitudes = location_data['Latitude (°)']
longitudes = location_data['Longitude (°)']
trail_coordinates = []

m = fo.Map(location=[65.059533, 25.472295], zoom_start=15)
fo.Marker([65.059533, 25.472295], popup='Linnanmaa').add_to(m)

for i in range(len(latitudes)):
    trail_coordinates.append([latitudes[i], longitudes[i]])

fo.PolyLine(trail_coordinates, tooltip="Polku", color='red').add_to(m)

m.save('path.html')

with open("path.html", "r", encoding="utf-8") as file:
    html_content = file.read()

st.components.v1.html(html_content, height=600, scrolling=True)

st.write("Map saved to path.html")

st.html('path.html')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

total_distance = 0
for i in range(len(latitudes) - 1):
    total_distance += haversine(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1])

st.write("Total distance: {:.2f} km".format(total_distance))

average_speed = total_distance / (location_data['Time (s)'].iloc[-1] / 3600)
st.write("Average speed: {:.2f} km/h".format(average_speed))

#Location and path plotting end

#Acceleration data filtering and plotting start
time = acceleration_data['Time (s)']
acceleration_x = acceleration_data['X (m/s^2)']
acceleration_y = acceleration_data['Y (m/s^2)']
acceleration_z = acceleration_data['Z (m/s^2)']

plt.title('Acceleration data')
plt.savefig('unfiltered_acceleration_data.png')
st.write("Unfiltered acceleration data saved to unfiltered_acceleration_data.png")
plt.show()
plt.close()

#Filtering
def butter_lowpass(cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

fs = 100
cutoff = 2
order = 6

x_filtered = butter_lowpass_filter(acceleration_x, cutoff, fs, order)
y_filtered = butter_lowpass_filter(acceleration_y, cutoff, fs, order)
z_filtered = butter_lowpass_filter(acceleration_z, cutoff, fs, order)

st.write("Acceleration data filtering done")

#Plotting filtered data

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(time, x_filtered, label='X')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
ax.grid()
st.pyplot(fig)

plt.savefig('filtered_acceleration_data.png')
plt.show()
st.write("Filtered acceleration data saved to filtered_acceleration_data.png")

#acceleration data filtering and plotting end

#PSD start
time = acceleration_data['Time (s)'].values
PSD_z_data = acceleration_data['Z (m/s^2)'].values

sampling_interval = np.mean(np.diff(time))
sampling_frequency = 1 / sampling_interval

nperseg = 256

frequencies, PSD_values = welch(PSD_z_data, fs=sampling_frequency, nperseg=nperseg)

fig, ax = plt.subplots(figsize=(10, 10))
plt.semilogy(frequencies, PSD_values)
ax.plot(frequencies, PSD_values, label='Z')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (m^2/s^3)')
ax.grid()
st.pyplot(fig)

plt.savefig('PSD.png')
plt.show()
st.write("PSD saved to PSD.png")
#PSD end

#Counting steps start

#Steps from z_filtered
steps = scipy.signal.find_peaks(z_filtered, height=0.25)[0].size
st.write("Step count ", steps)

#Fourier analysis start
time = acceleration_data['Time (s)'].values
Fourier_z_filtered = acceleration_data['Z (m/s^2)'].values
N = len(Fourier_z_filtered)
positive = Fourier_z_filtered > 0
positive_fft = np.abs(fft(z_filtered[positive]))

interval = np.mean(np.diff(time))
freq = 1 / interval
n = len(z_filtered)
frequencies = fftfreq(n, interval)
fft_values = fft(Fourier_z_filtered)

positive_frequencies = frequencies[:n // 2]
positive_fft = np.abs(fft_values[:n // 2])

min_frequency = 0.5
max_frequency = 3

valid_indices = (positive_frequencies >= min_frequency) & (positive_frequencies <= max_frequency)
dominant_frequency = positive_frequencies[valid_indices][np.argmax(positive_fft[valid_indices])]

total_time = time[-1] - time[0]
fourier_steps = total_time * dominant_frequency
st.write("Fourier step count ", int(fourier_steps))
#Fourier analysis end
#Counting steps end

#steps lenght
total_distance_meters = total_distance * 1000
step_length = total_distance_meters / steps
st.write("Average step length: {:.2f} m".format(step_length))