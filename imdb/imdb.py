from keras.datasets import imdb
from keras import models, layers, optimizers, losses
import numpy as np
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def main():
    (train_data, train_labels), (test_data, test_labels) \
        = imdb.load_data(num_words=10000)

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                  loss=losses.mse,
                  metrics=['acc'])
    history = model.fit(partial_x_train,
                        partial_y_train,
                        batch_size=512,
                        epochs=4,
                        validation_data=(x_val, y_val))
    results = model.evaluate(x_test, y_test)
    print(results)

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training on validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
