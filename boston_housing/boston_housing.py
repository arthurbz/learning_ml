from keras.datasets import boston_housing
from keras import models, layers
import numpy as np


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimer='rmsprop', loss='mse', metrics=['mae'])
    return model


if __name__ == '__main__':
    (train_data, train_targets), (test_data, test_targets) = \
            boston_housing.load_data()

    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []
