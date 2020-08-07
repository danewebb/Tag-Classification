import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

set_epochs = 10
set_valsteps = 20
embedding_dim = 16
dense1_size = 16
dense2_size = 1



train_data, test_data, info = tfds.load(
    'scientific_papers',
    split=['train', 'test'],
    as_supervised=True, with_info=True
)

encoder = info.features['text'].encoder


train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes=([None], ()))
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes=([None], ()))

model = keras.Sequential([
    layers.Embedding(encoder.vocab_size, embedding_dim),
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
    validation_data=test_batches,
    validation_steps=set_valsteps
)

