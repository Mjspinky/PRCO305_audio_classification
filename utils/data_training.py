import librosa
import pandas as pd
import numpy as np

import os
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Keras
from keras import models, layers
from tensorflow.keras.models import load_model
import ffmpeg
import glob


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


def generating_dataset():
    # generating a dataset

    data_goes_here = 'processed_data/data.csv'
    create_data_file(data_goes_here)
    # TODO:Remove a few of these from the the list in some tests(It will take a while) once you've got some
    # decent amounts of confirmed latin musics
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock salsa'.split()
    for g in genres:
        for filename in os.listdir(f'./genres/{g}'):
            # TODO:Research what each of these mean and do, potentially add in one or two more of the librosa number
            #  generators
            songname = f'./genres/{g}/{filename}'
            create_data(songname, filename, g, data_goes_here)


def dataset_training():
    # reading dataset from csv
    print("reading Dataset")
    data = pd.read_csv('../processed_data/data.csv')
    data.head()

    # Dropping unneccesary columns
    print("dropping unnessesary columns")
    data = data.drop(['filename'], axis=1)
    data.head()

    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    print(y)
    # normalizing
    print("normalising")
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

    # spliting of dataset into train and test dataset
    print("splitting dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # TODO:test different activation functions on different layers, Add and subtract layers. Make sure to research.
    # creating a model
    print("creating model")
    model = models.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(X_train.shape[1],)))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(11, activation='sigmoid'))  # has to be above the number of Genres being tested

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train,
                        y_train,
                        epochs=50,
                        batch_size=128)

    # calculate accuracy
    print("Calculate accuracy")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('test_acc: ', test_acc)
    model.save("models/data_model")
    # predictions
    predictions = model.predict(X_test)
    print(np.argmax(predictions[0]))


def recorded_data_preparation():
    try:
        dir_name = ".."
        test = os.listdir(dir_name)
        for item in test:
            if item.endswith(".wav"):
                for file in test:
                    if file.endswith(".au"):
                        os.remove(os.path.join(dir_name, file))

                # File conversion
                stream = ffmpeg.input('current_recording.wav')
                stream = ffmpeg.output(stream, 'current_recording.au')
                ffmpeg.run(stream)
                # if there is no .wav file it will break before this point, so the rest of this is only ever done if its got new
                # data to process
                data_goes_here = 'processed_data/recorded_data.csv'
                create_data_file(data_goes_here)

                song_name = 'current_recording.au'
                create_data(song_name, song_name, 'test', data_goes_here)

                os.remove(os.path.join(dir_name, item))
    except:
        print("No new recording, Skipping new data preparation")




def model_predict():
    model = load_model("../models/data_model/")

    extension = 'csv'
    all_filenames = [i for i in glob.glob('processed_data/*.{}'.format(extension))]

    # combine all files in the list
    combined_data = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_data.to_csv("processed_data/combined_csv.csv", index=False, encoding='utf-8-sig')

    data = pd.read_csv('processed_data/combined_csv.csv')
    os.remove(os.path.join('../processed_data/', 'combined_csv.csv'))
    data.head()

    print("dropping unnessesary columns")
    data = data.drop(['filename'], axis=1)
    data.head()

    print("normalising the data")
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
    X = normalized_data[-1]
    test = [float(i) for i in X]
    predictions = model.predict([test])
    print(represent_prediction(np.argmax(predictions[0])))
    return represent_prediction(np.argmax(predictions[0]))

def represent_prediction(prediction):
    genre_list = {
        0: "Blues",
        1: "Classical",
        2: "Country",
        3: "Disco",
        4: "Hiphop",
        5: "Jazz",
        6: "Metal",
        7: "Pop",
        8: "Reggae",
        9: "Rock",
        10: "Salsa",
    }
    return genre_list.get(prediction, "Cannot determine the right style")


# TODO:Tensorflow tests?
# dataset_training()
# test_csv_data_creation()
model_predict()
# test_data_preparation()
