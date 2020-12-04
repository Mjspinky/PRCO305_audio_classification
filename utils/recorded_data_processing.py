import pandas as pd
import numpy as np
import os
# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Keras
from tensorflow.keras.models import load_model
import ffmpeg
import glob

from utils.data_processing_utils import create_data_file, create_data

def recorded_data_preparation():
    # try:
    dir_name = "./utils"
    test = os.listdir(dir_name)
    for item in test:
        if item.endswith(".wav"):
            for file in test:
                if file.endswith(".au"):
                    os.remove(os.path.join(dir_name, file))
            print("hello")
            # File conversion
            stream = ffmpeg.input('utils/current_recording.wav')
            stream = ffmpeg.output(stream, 'utils/current_recording.au')
            ffmpeg.run(stream)
            # if there is no .wav file it will break before this point, so the rest of this is only ever done if its got new
            # data to process;
            data_goes_here = './processed_data/recorded_data.csv'
            create_data_file(data_goes_here)

            song_name = 'utils/current_recording.au'
            create_data(song_name, song_name, 'test', data_goes_here)

            os.remove(os.path.join(dir_name, item))
    # except:
    #    print("No new recording, Skipping new data preparation")
    return model_predict()




def model_predict():
    model = load_model("../models/data_model/")

    extension = 'csv'
    all_filenames = [i for i in glob.glob('./processed_data/*.{}'.format(extension))]

    # combine all files in the list
    combined_data = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_data.to_csv("./processed_data/combined_csv.csv", index=False, encoding='utf-8-sig')

    data = pd.read_csv('./processed_data/combined_csv.csv')
    os.remove(os.path.join('./processed_data/', 'combined_csv.csv'))
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

# recorded_data_preparation()
# model_predict()
