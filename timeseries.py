import os
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sound_to_serie_temporelle(datafolder):
    signals_normal=[]
    signals_anomaly=[]
    signals_normal_aug_1=[]
    signals_anomaly_aug_1=[]
    signals_normal_aug_2=[]
    signals_anomaly_aug_2=[]
    audio_files = [os.path.join(root, file) for root, _, files in os.walk(datafolder) for file in files if file.endswith('.wav')]
    for audio_file in audio_files:
        audio_data, sample_rate = sf.read(audio_file)
        audio_data = get_normalized_audio(audio_data)
        if 'anomaly' in audio_file.lower():
            if 'aug1' in os.path.basename(audio_file):
                signals_anomaly_aug_1.append((audio_data, os.path.basename(audio_file)))
            elif 'aug2' in os.path.basename(audio_file) :
                signals_anomaly_aug_2.append((audio_data, os.path.basename(audio_file)))
            else:
                signals_anomaly.append((audio_data, os.path.basename(audio_file)))
        else:
            if 'aug1' in os.path.basename(audio_file):
                signals_normal_aug_1.append((audio_data, os.path.basename(audio_file)))
            elif 'aug2' in os.path.basename(audio_file) :
                signals_normal_aug_2.append((audio_data, os.path.basename(audio_file)))
            else:
                signals_normal.append((audio_data, os.path.basename(audio_file)))
    return signals_normal, signals_normal_aug_1, signals_normal_aug_2,signals_anomaly,signals_anomaly_aug_1, signals_anomaly_aug_2

def get_normalized_audio(audio_data):
    normalized_audio = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))
    return normalized_audio

def apply_window(audio_data, window_size, overlap=0):
    segments = []
    start = 0
    while start + window_size <= len(audio_data):
        segment = audio_data[start:start + window_size]
        segment=get_normalized_audio(segment)
        segments.append(segment)
        if overlap!=0:
            start += window_size-overlap
        else:
            start += window_size
    return segments

def apply_window_all(signals_normal,all_segments_normal, window_size, overlap=0):
    for signal_data, _ in signals_normal:
        segments = apply_window(signal_data, window_size, overlap)
        all_segments_normal.append(segments)
    return all_segments_normal


def create_dataframes(all_segments_normal, all_segments_anomaly, 
                      all_segments_normal_aug_1, all_segments_anomaly_aug_1, 
                      all_segments_normal_aug_2, all_segments_anomaly_aug_2):
    # Create a DataFrame for normal segments
    df_normal = pd.DataFrame(np.concatenate(all_segments_normal))
    df_normal.to_csv("C:/Users/dorbe/Desktop/PI2_2/csv_data/normal_segments.csv", index=False)

    # Create a DataFrame for anomaly segments
    df_anomaly = pd.DataFrame(np.concatenate(all_segments_anomaly))
    df_anomaly.to_csv("C:/Users/dorbe/Desktop/PI2_2/csv_data/anomaly_segments.csv", index=False)

    # Create a DataFrame for normal segments with augmentation 1
    df_normal_aug_1 = pd.DataFrame(np.concatenate(all_segments_normal_aug_1))
    df_normal_aug_1.to_csv("C:/Users/dorbe/Desktop/PI2_2/csv_data/normal_segments_aug_1.csv", index=False)

    # Create a DataFrame for anomaly segments with augmentation 1
    df_anomaly_aug_1 = pd.DataFrame(np.concatenate(all_segments_anomaly_aug_1))
    df_anomaly_aug_1.to_csv("C:/Users/dorbe/Desktop/PI2_2/csv_data/anomaly_segments_aug_1.csv", index=False)

    # Create a DataFrame for normal segments with augmentation 2
    df_normal_aug_2 = pd.DataFrame(np.concatenate(all_segments_normal_aug_2))
    df_normal_aug_2.to_csv("C:/Users/dorbe/Desktop/PI2_2/csv_data/normal_segments_aug_2.csv", index=False)

    # Create a DataFrame for anomaly segments with augmentation 2
    df_anomaly_aug_2 = pd.DataFrame(np.concatenate(all_segments_anomaly_aug_2))
    df_anomaly_aug_2.to_csv("C:/Users/dorbe/Desktop/PI2_2/csv_data/anomaly_segments_aug_2.csv", index=False)


# Test of time series transformation
datafolder = 'C:/Users/dorbe/Desktop/PI2_2/Valmet'
signals_normal=[]
signals_anomaly=[]
signals_normal_aug_1=[]
signals_anomaly_aug_1=[]
signals_normal_aug_2=[]
signals_anomaly_aug2=[]
signals_normal, signals_normal_aug_1, signals_normal_aug_2,signals_anomaly,signals_anomaly_aug_1, signals_anomaly_aug_2=sound_to_serie_temporelle(datafolder)
#Displaying an example of a normal signal 
plt.figure(figsize=(10, 4))
audio_data, filename = signals_normal[45]
time_axis = np.arange(len(audio_data)) / 96000
plt.plot(time_axis,audio_data)
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.title("Signal normal: " + filename)
plt.show()

# Displaying an example of a signal with anomaly
plt.figure(figsize=(10, 4))
audio_data, filename = signals_anomaly[2]
time_axis = np.arange(len(audio_data)) / 96000
plt.plot(time_axis,audio_data)
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.title("Signal with anomaly: " + filename)
plt.show()

# Apply the window to all audio files
all_segments_normal = []
all_segments_anomaly = []
all_segments_normal_aug_1 = []
all_segments_anomaly_aug_1 = []
all_segments_normal_aug_2 = []
all_segments_anomaly_aug_2 = []
all_segments_normal = apply_window_all(signals_normal,all_segments_normal, window_size=1000, overlap=0)
all_segments_normal_aug1 = apply_window_all(signals_normal_aug_1,all_segments_normal_aug_1, window_size=1000, overlap=0)
all_segments_normal_aug2 = apply_window_all(signals_normal_aug_2,all_segments_normal_aug_2, window_size=1000, overlap=0)
all_segments_anomaly = apply_window_all(signals_anomaly,all_segments_anomaly,window_size=1000, overlap=0)
all_segments_anomaly_aug1 = apply_window_all(signals_anomaly_aug_1,all_segments_anomaly_aug_1,window_size=1000, overlap=0)
all_segments_anomaly_aug2 = apply_window_all(signals_anomaly_aug_2,all_segments_anomaly_aug_2,window_size=1000, overlap=0)
index=0
# Plot all segments of the first audio file
for i, segment in enumerate(all_segments_normal[index]):
  plt.figure()
  time_axis = np.arange(len(segment)) / 96000
  plt.plot(time_axis,segment)
  plt.xlabel("Time")
  plt.ylabel("Amplitude")
  plt.title(f"Segment {i+1} of the {index+1} Audio Signal")
  plt.show()
create_dataframes(all_segments_normal, all_segments_anomaly, all_segments_normal_aug_1, all_segments_anomaly_aug_1, all_segments_normal_aug_2, all_segments_anomaly_aug_2)
