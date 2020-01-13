import tensorflow as tf
import tensorboard as tb
import numpy as np
import pickle
import os
import warnings
import datetime
from tensorflow.python.framework import ops
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.classification import accuracy_score






def process_data(*args):
    for dir in args:



# ignore all of the tensorflow warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()


# n_classes?? how many different tags?
# [R, V, I, L, C, B, J, S, A, D, T]
n_classes = 11


train_data_dir = 'training_dict.pkl'
test_data_dir = 'testing_dict.pkl'
vocab_dir = 'vocab.pkl'
with open('training_dict.pkl', 'rb') as f1:
    train_dict = pickle.load(f1)

with open('testing_dict.pkl', 'rb') as f2:
    test_dict = pickle.load(f2)

with open('rank_vocab.pkl', 'rb') as f3:
    vocab_dict = pickle.load(f3)



vocab_size = len(vocab_dict)
# do we 'unk' vocab?

# process training data
for key, value in train_dict.items():
    train_x = value['paragraph']

n_samples= None # Set n_samples=None to use the whole dataset

summaries_dir= 'logs/'# Directory where TensorFlow summaries will be stored'
batch_size = 100 #Batch size
train_steps = 1000 #Number of training steps
hidden_size= 75 # Hidden size of LSTM layer
embedding_size = 75 # Size of embeddings layer

random_state = 24 # Random state used for data splitting. Default is 0

learning_rate = 0.01
test_size = 0.2
dropout_keep_prob = 0.5 # 0<dropout_keep_prob<=1. Dropout keep-probability
sequence_len = None # Maximum sequence length
validate_every = 100 # Step frequency in order to evaluate the model using a validation set'

# Prepare summaries
summaries_dir = '{0}/{1}'.format(summaries_dir, datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'))
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

# Prepare model directory
model_name = str(int(time.time()))
model_dir = '{0}/{1}'.format('checkpoints', model_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)



lstm_model = LSTM_RNN(hidden_size, vocab_size, embedding_size, n_classes, max_len=sequence_len,
                      learning_rate=learning_rate, rand_state=random_state)


sess = tf.Session()
sess.run(tf.global_variables_initializer)

saver = tf.Saver()
train_writer.add_graph(lstm_model.input.graph)




