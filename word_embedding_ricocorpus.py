import tensorflow as tf
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

import pickle


def padding(data, data_len, maxlen=0):
    old_num = 0
    if maxlen == 0:
        # find necessary padding length
        for num in data_len:
            if num > old_num:
                old_num = num

        padlen = old_num
    else:
        padlen = maxlen

    for arr in data:
        while len(arr) < padlen:
            arr.append(0)

    return data

batch_size = 20
set_epochs = 10
set_valsteps = 20
embedding_dim = 16
dense1_size = 16
dense2_size = 1

with open('train_vec.pkl', 'rb') as file:
    train_data = pickle.load(file)

with open('train_len.pkl', 'rb') as file1:
    train_lens = pickle.load(file1)


    
# train_data_arr = np.array([np.array(x) for x in train_data])

with open('vocab.pkl', 'rb') as voc_file:
    vocab = pickle.load(voc_file)

vocab_size = len(vocab)


padded_train_data = padding(train_data, train_lens)

# csv path

# with open('train_data.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(padded_train_data)
#
# train_csv_data = tf.data.experimental.make_csv_dataset(
#     padded_train_data,
#     batch_size,
#     num_epochs=set_epochs
# )


# non-csv path
train_data_arr = np.array([np.array(x) for x in padded_train_data])



train_dataset = tf.data.Dataset.from_tensors(train_data_arr)


#convert train_vec into tf



model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim),
    layers.GlobalAveragePooling1D(),
    layers.Dense(dense1_size, activation='relu'),
    layers.Dense(dense2_size)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_batches,
    epochs=set_epochs,
)







