import pandas as pd
import numpy as np
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Keras
from keras import models, layers


def dataset_training():
    # reading dataset from csv
    print("reading Dataset")
    data = pd.read_csv('processed_data/data.csv')
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
    model.save("../models/data_model")
    # predictions
    predictions = model.predict(X_test)
    print(np.argmax(predictions[0]))


dataset_training()
