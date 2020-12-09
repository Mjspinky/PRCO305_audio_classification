import librosa
import numpy as np
import os
import csv


def create_header():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate tempo'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    return header


def create_data_file(data_goes_here):
    file = open(data_goes_here, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(create_header())


def create_data(song_name, file_name, g, data_goes_here):
    y, sr = librosa.load(song_name, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tempo = librosa.feature.tempogram(y=y, sr=sr)  # Tempo is a big thing for latin music
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)  # This is 100% useful

    to_append = f'{file_name} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(tempo)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    to_append += f' {g}'
    file = open(data_goes_here, 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())


def generating_dataset():  # this takes 30+ minutes to run, run this at your own caution
    # generating a dataset

    data_goes_here = '../processed_data/data.csv'
    create_data_file(data_goes_here)
    genres = 'bachata cha_cha kizomba salsa'.split()
    for g in genres:
        for filename in os.listdir(f'../genres/{g}'):
            songname = f'../genres/{g}/{filename}'
            create_data(songname, filename, g, data_goes_here)

# generating_dataset()
