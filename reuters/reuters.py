from keras.datasets import reuters
import numpy as np


def vectorize_sequences(sequences, dimension=10):
    '''results is a matrix with len(sequences) positions,
    and every single position is an array of 10000 positions
    where we can use 1 or 0 to know wheter a word was present
    in the review or not
    '''
    results = np.zeros((len(sequences), dimension))
    print(list(enumerate(sequences)))
    for i, sequence in enumerate(sequences):
        print(i)
        print(sequence)
        print(list(results[i, sequence]))
        results[i, sequence] = 1
        print(list(results[i, sequence]))
    print(results)
    return results


'''(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data(num_words=10000)'''

train_data = [[2, 4, 6, 7, 2, 3], [0, 2, 4, 8], [3, 6, 8], [2, 4, 6, 7]]

x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)

# print(len(train_data)) == 8982
# print(len(test_data)) == 2246
